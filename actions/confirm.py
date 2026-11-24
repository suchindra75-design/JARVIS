"""
actions/confirm.py — human-in-the-loop approval for risky tool actions.

The model can be steered by text it reads (web pages, screenshots, copied text,
files), so anything that runs code, sends messages, changes files or powers off
the machine must be approved by the person at the keyboard. The approval
happens in the desktop UI, never through a tool argument the model can set.

Fails closed: with no UI, an error or no answer before the timeout, the
action is refused.
"""

from __future__ import annotations

DEFAULT_TIMEOUT = 60  # seconds before an unanswered request counts as "Deny"

DECLINED = (
    "The user declined this action in the on-screen confirmation, or did not answer in time. "
    "It was NOT performed. Tell the user briefly and do not retry unless they ask again."
)


def request_confirmation(player, title: str, detail: str = "",
                         timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Ask the user to approve an action. Blocks the calling worker thread.

    `player` is the RayaUI instance passed to every action. Returns True only
    when the user clicks Allow.
    """
    ask = getattr(player, "confirm_action", None)
    if ask is None:
        print(f"[Confirm] No UI available — refusing: {title}")
        return False
    try:
        return bool(ask(title, detail, timeout))
    except Exception as e:
        print(f"[Confirm] Confirmation failed ({e}) — refusing: {title}")
        return False


def preview(text, limit: int = 600) -> str:
    """Shorten long values for display in the confirmation panel."""
    s = str(text if text is not None else "").strip()
    return s if len(s) <= limit else s[:limit] + f"\n… ({len(s) - limit} more characters)"
