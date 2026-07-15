from pathlib import Path
from core.tracking.dasiamrpn_wrapper import DaSiamRPNTracker

TRACKER_MODEL_PATH = Path("weights/SiamRPNOTB.model")

_tracker: DaSiamRPNTracker | None = None


def _build_tracker() -> DaSiamRPNTracker:
    """
    Creates a tracker instance using the official FalconEye checkpoint.
    """
    if not TRACKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tracker checkpoint not found: '{TRACKER_MODEL_PATH}'.\n"
            "Download the DaSiamRPN OTB checkpoint from the latest GitHub Release "
            "and place it in the 'weights/' directory."
        )

    return DaSiamRPNTracker(model_path=str(TRACKER_MODEL_PATH))


def create_tracker() -> DaSiamRPNTracker:
    """
    Creates a brand-new tracker instance,
    replacing any existing tracker.
    """
    global _tracker

    _tracker = _build_tracker()
    return _tracker


def get_tracker() -> DaSiamRPNTracker:
    """
    Returns the current tracker instance.
    Lazily creates one if it doesn't already exist.
    """
    global _tracker

    if _tracker is None:
        _tracker = _build_tracker()

    return _tracker


def reset_tracker() -> None:
    """
    Removes the current tracker instance.
    """
    global _tracker
    _tracker = None