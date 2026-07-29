from pathlib import Path
from core.tracking.dasiamrpn_wrapper import DaSiamRPNTracker

TRACKER_MODEL_PATH = Path("weights/SiamRPNOTB.model")

def create_tracker() -> DaSiamRPNTracker:
    """
    Each caller receives an independent tracker with its own internal state,
    making it suitable for per-session usage in FastAPI/WebSocket handlers.
    """
    if not TRACKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tracker checkpoint not found: '{TRACKER_MODEL_PATH}'.\n"
            "Download the DaSiamRPN OTB checkpoint from the latest GitHub Release "
            "and place it in the 'weights/' directory."
        )

    return DaSiamRPNTracker(model_path=str(TRACKER_MODEL_PATH))