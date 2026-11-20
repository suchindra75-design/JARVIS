"""
audio/speaker_id.py — local voice identity ("is this the enrolled user?").

What it is: a speaker-embedding model (WeSpeaker ResNet34, run by sherpa-onnx)
turns ≥1 s of speech into a 256-number voiceprint; we compare it with the
enrolled profile by cosine similarity.

What it is NOT: security. Recordings and similar voices can fool it, so it is
display-only — it never approves actions and never decides what JARVIS does.
It is separate from transcription (that's Gemini, and says *what* was said).

Privacy: the profile (voiceprints only, no audio) is stored in
memory/voice_profile.npz with 0600 permissions, ignored by git, and never
sent anywhere. delete_profile() removes it.

Threads: enrol/verify calls do heavy work (~180 ms per 3 s of audio). Use
`submit()` to verify off the audio and UI threads; it runs on one low-priority
worker with a 2-item queue (older requests are dropped, never piled up).
"""

from __future__ import annotations

import collections
import hashlib
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/" + MODEL_NAME)
MODEL_SHA256 = "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"
MODEL_MB = 26.5
SAMPLE_RATE = 16000

ENROL_SAMPLES = 5                 # phrases recorded during enrolment
MIN_SPEECH_S = 1.0                # shorter utterances are "too short" (not guessed)
ENROL_MIN_SPEECH_S = 2.0
THRESHOLD_FLOOR = 0.45            # never accept below this, whatever enrolment says
THRESHOLD_CEIL = 0.90
UNCERTAIN_BAND = 0.08             # just below the threshold = "uncertain"

ENROL_PHRASES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Please remind me to water the plants every Sunday morning.",
    "My favourite season is autumn because the evenings are calm.",
    "Open the calendar and show me what is planned for next week.",
    "I usually drink a cup of tea before I start working.",
]


@dataclass
class Verdict:
    label: str                    # "you" | "uncertain" | "unknown" | "too_short" | "not_enrolled" | "unavailable"
    score: float | None = None
    threshold: float | None = None
    speech_s: float = 0.0
    at: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        return {
            "you": "You", "uncertain": "Not sure", "unknown": "Unknown speaker",
            "too_short": "Not enough speech", "not_enrolled": "Voice not enrolled",
            "unavailable": "Voice identity unavailable",
        }[self.label]


@dataclass
class SampleCheck:
    ok: bool
    reason: str = ""
    speech_s: float = 0.0
    level_dbfs: float = -120.0


def _speech_only(x: np.ndarray, frame: int = 320) -> np.ndarray:
    """Drop near-silent 20 ms frames (relative to the loudest part), keep the rest."""
    n = len(x) // frame
    if n == 0:
        return x[:0]
    fr = x[: n * frame].reshape(n, frame)
    e = 10 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)
    keep = e > max(e.max() - 35.0, -60.0)
    return fr[keep].reshape(-1)


class SpeakerIdentity:
    def __init__(self, model_dir: Path | None = None, profile_path: Path | None = None):
        self.model_path = Path(model_dir or ROOT / "models") / MODEL_NAME
        self.profile_path = Path(profile_path or ROOT / "memory" / "voice_profile.npz")
        self._ext = None
        self._ext_lock = threading.Lock()
        self.error: str | None = None
        self.profile: dict | None = None
        self.last: Verdict | None = None
        self._q: collections.deque = collections.deque(maxlen=2)
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self._load_profile()

    # ── availability ─────────────────────────────────────────────────────

    @staticmethod
    def library_available() -> bool:
        try:
            import sherpa_onnx  # noqa: F401
            return True
        except Exception:
            return False

    @property
    def model_present(self) -> bool:
        return self.model_path.exists()

    @property
    def status(self) -> str:
        """unavailable | no_model | not_enrolled | ready"""
        if not self.library_available() or self.error:
            return "unavailable"
        if not self.model_present:
            return "no_model"
        if self.profile is None:
            return "not_enrolled"
        return "ready"

    def download_model(self, progress=None) -> None:
        """Download the speaker model (≈26.5 MB) and verify its checksum. Blocking."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".part")
        h = hashlib.sha256()
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk); h.update(chunk); got += len(chunk)
                if progress and total:
                    progress(got / total)
        if h.hexdigest() != MODEL_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Downloaded model failed its checksum and was discarded.")
        os.replace(tmp, self.model_path)

    def _extractor(self):
        if self._ext is None:
            import sherpa_onnx
            if not self.model_present:
                raise RuntimeError("Speaker model not downloaded.")
            with open(self.model_path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != MODEL_SHA256:
                    raise RuntimeError("Speaker model file is corrupt (checksum mismatch).")
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model_path), num_threads=1, provider="cpu")
            self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        return self._ext

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """L2-normalised voiceprint of float32 16 kHz mono audio."""
        x = np.asarray(samples, dtype=np.float32)
        if x.dtype != np.float32 or (len(x) and np.max(np.abs(x)) > 4.0):
            x = x.astype(np.float32) / 32768.0                     # int16 given
        with self._ext_lock:
            ext = self._extractor()
            s = ext.create_stream()
            s.accept_waveform(SAMPLE_RATE, x)
            s.input_finished()
            e = np.array(ext.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    # ── enrolment ────────────────────────────────────────────────────────

    @staticmethod
    def check_sample(samples: np.ndarray, min_speech_s: float = ENROL_MIN_SPEECH_S) -> SampleCheck:
        """Is this recording good enough to enrol? Returns a reason a person can act on."""
        x = np.asarray(samples, dtype=np.float32)
        if len(x) and np.max(np.abs(x)) > 4.0:
            x = x / 32768.0
        if len(x) == 0:
            return SampleCheck(False, "Nothing was recorded. Check the microphone.")
        clip = float(np.mean(np.abs(x) > 0.99))
        sp = _speech_only(x)
        speech_s = len(sp) / SAMPLE_RATE
        lvl = 10 * np.log10(float(np.mean(sp * sp)) + 1e-12) if len(sp) else -120.0
        if clip > 0.002:
            return SampleCheck(False, "Too loud — it clipped. Move back a little or lower the input level.",
                               speech_s, lvl)
        if lvl < -50:
            return SampleCheck(False, "Too quiet. Speak up or move closer to the mic.", speech_s, lvl)
        if speech_s < min_speech_s:
            return SampleCheck(False, f"Not enough speech ({speech_s:.1f} s). Read the whole phrase.", speech_s, lvl)
        # rough noise check: quiet frames vs speech frames
        n = len(x) // 320
        if n > 10:
            e = np.sort(10 * np.log10(np.mean(x[: n * 320].reshape(n, 320) ** 2, axis=1) + 1e-12))
            if e[-n // 5:].mean() - e[: n // 5].mean() < 12:
                return SampleCheck(False, "Too much background noise. Try somewhere quieter.", speech_s, lvl)
        return SampleCheck(True, "", speech_s, lvl)

    def enrol(self, samples: list[np.ndarray], append: bool = False) -> dict:
        """Build (or extend) the profile from recordings that passed check_sample()."""
        embs = [self.embed(s) for s in samples]
        if append and self.profile is not None:
            embs = list(self.profile["embeddings"]) + embs
        if len(embs) < 3:
            raise ValueError("At least 3 good samples are needed.")
        E = np.stack(embs)
        centroid = E.mean(0)
        centroid /= np.linalg.norm(centroid)
        # leave-one-out: how well does each sample match the others?
        loo = []
        for i in range(len(E)):
            c = np.delete(E, i, 0).mean(0)
            loo.append(float(E[i] @ (c / np.linalg.norm(c))))
        # Threshold sits just below your least-typical sample. The margin grows
        # with how much your samples vary (consistent voice → tighter match).
        margin = float(np.clip(3 * np.std(loo), 0.06, 0.15))
        thr = float(np.clip(min(loo) - margin, THRESHOLD_FLOOR, THRESHOLD_CEIL))
        self.profile = {"embeddings": E, "centroid": centroid, "threshold": thr,
                        "created": time.time() if not (append and self.profile) else self.profile["created"],
                        "updated": time.time(), "model": MODEL_NAME, "loo": np.array(loo)}
        self._save_profile()
        return {"samples": len(E), "threshold": thr, "consistency": float(np.mean(loo))}

    # ── verification ─────────────────────────────────────────────────────

    def verify(self, samples: np.ndarray) -> Verdict:
        st = self.status
        if st == "unavailable" or st == "no_model":
            return Verdict("unavailable")
        if st == "not_enrolled":
            return Verdict("not_enrolled")
        x = np.asarray(samples, dtype=np.float32)
        if len(x) and np.max(np.abs(x)) > 4.0:
            x = x / 32768.0
        sp = _speech_only(x)
        speech_s = len(sp) / SAMPLE_RATE
        if speech_s < MIN_SPEECH_S:
            return Verdict("too_short", speech_s=speech_s)
        try:
            score = float(self.embed(sp) @ self.profile["centroid"])
        except Exception as e:
            self.error = str(e)
            return Verdict("unavailable")
        thr = self.profile["threshold"]
        label = "you" if score >= thr else ("uncertain" if score >= thr - UNCERTAIN_BAND else "unknown")
        return Verdict(label, score, thr, speech_s)

    def submit(self, samples: np.ndarray, on_result) -> None:
        """Verify in the background; on_result(Verdict) is called on the worker thread."""
        if self.status != "ready":
            return
        self._q.append((samples, on_result))
        self._wake.set()
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="SpeakerID", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        try:
            os.nice(5)                                  # lower than audio/UI work (POSIX only)
        except Exception:
            pass
        while True:
            if not self._wake.wait(30):
                return                                  # idle: let the thread end
            self._wake.clear()
            while self._q:
                samples, cb = self._q.popleft()
                v = self.verify(samples)
                self.last = v
                try:
                    cb(v)
                except Exception:
                    pass

    # ── profile file ─────────────────────────────────────────────────────

    @property
    def summary(self) -> dict:
        p = self.profile
        if not p:
            return {}
        return {"samples": int(len(p["embeddings"])), "threshold": float(p["threshold"]),
                "created": float(p["created"]), "updated": float(p["updated"]),
                "consistency": float(np.mean(p["loo"]))}

    def _save_profile(self) -> None:
        p = self.profile
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.profile_path.with_name(self.profile_path.stem + ".tmp.npz")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            np.savez(f, embeddings=p["embeddings"], centroid=p["centroid"], threshold=p["threshold"],
                     created=p["created"], updated=p["updated"], loo=p["loo"], model=np.array(p["model"]))
        os.replace(tmp, self.profile_path)
        os.chmod(self.profile_path, 0o600)

    def _load_profile(self) -> None:
        if not self.profile_path.exists():
            self.profile = None
            return
        try:
            with np.load(self.profile_path, allow_pickle=False) as d:
                if str(d["model"]) != MODEL_NAME:
                    raise ValueError("profile was made with a different model")
                self.profile = {k: d[k] for k in ("embeddings", "centroid", "loo")}
                self.profile.update(threshold=float(d["threshold"]), created=float(d["created"]),
                                    updated=float(d["updated"]), model=str(d["model"]))
        except Exception as e:
            print(f"[SpeakerID] Ignoring unreadable voice profile: {e}")
            self.profile = None

    def delete_profile(self) -> None:
        self.profile = None
        self.last = None
        try:
            self.profile_path.unlink()
        except FileNotFoundError:
            pass
