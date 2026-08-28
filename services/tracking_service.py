from huggingface_hub import hf_hub_download

from core.tracking.dasiamrpn_wrapper import DaSiamRPNTracker

HF_REPO_ID = "Varun-Sai-500/DaSiamRPN"
HF_FILENAME = "SiamRPNOTB.model"

def create_tracker() -> DaSiamRPNTracker:
    """
    Each caller receives an independent tracker with its own internal state,
    making it suitable for per-session usage in FastAPI/WebSocket handlers.
    """
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
    )

    return DaSiamRPNTracker(model_path=model_path)