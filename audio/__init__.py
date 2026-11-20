"""Audio capture, voice processing and playback for JARVIS.

engine.py  — AudioEngine: devices, streams, buffers, DSP worker thread
voice.py   — VoiceProcessor: echo cancellation, noise suppression, gain, VAD
output.py  — OutputEnhancer: loudness normalisation, limiter, volume ramps, ducking
"""
