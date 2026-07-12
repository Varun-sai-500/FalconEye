from core.tracking.dasiamrpn_wrapper import DaSiamRPNTracker

_tracker: DaSiamRPNTracker | None = None

def create_tracker() -> DaSiamRPNTracker:
    """
    Creates a brand-new tracker instance,
    replacing any existing tracker.
    """
    global _tracker

    _tracker = DaSiamRPNTracker(model_path="weights/SiamRPNVOT.model")
    return _tracker


def get_tracker() -> DaSiamRPNTracker:
    """
    Returns the current tracker instance.
    Lazily creates one if it doesn't already exist.
    """
    global _tracker

    if _tracker is None:
        _tracker = DaSiamRPNTracker(model_path="weights/SiamRPNVOT.model")

    return _tracker


def reset_tracker() -> None:
    """
    Removes the current tracker instance.
    """
    global _tracker
    _tracker = None
