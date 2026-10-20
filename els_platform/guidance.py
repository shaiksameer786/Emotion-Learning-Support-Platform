"""Local, non-generative learning guidance for the application shell."""

from __future__ import annotations


def get_local_guidance() -> tuple[str, ...]:
    """Return practical general tips without inspecting or retaining learner text."""
    return (
        "Write down the exact question or step that is blocking your progress.",
        "Try one focused 20-minute study block, then take a short break.",
        "Explain the concept in your own words or work through one small example.",
        "If you are still stuck, ask a teacher, tutor, or classmate a specific question.",
    )
