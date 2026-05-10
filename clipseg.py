import torch
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sam_model import call_sam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load CLIPSeg ---
MODEL_ID = "CIDAS/clipseg-rd64-refined"
processor = CLIPSegProcessor.from_pretrained(
    MODEL_ID,
    backend="torchvision"
)
model = CLIPSegForImageSegmentation.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

model.eval()

# --- Main segmentation function ---
def clipping(frame: np.ndarray, ref_image: np.ndarray = None, text: str = None) -> np.ndarray:
    if (ref_image is None) and (text is None):
        raise ValueError("Provide either a reference image or a text prompt.")

    frame = frame.astype(np.uint8)
    image_pil = Image.fromarray(frame)
    original_h, original_w = frame.shape[:2]

    # --- Reference image mode ---
    if ref_image is not None:
        ref_pil = Image.fromarray(ref_image.astype(np.uint8))
        cond = processor(images=ref_pil, return_tensors="pt")["pixel_values"].to(device)

        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        inputs["conditional_pixel_values"] = cond

    else:  # text is not None
        inputs = processor(images=image_pil, text=[text], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- Model inference ---
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        resized_logits = F.interpolate(
            logits,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )

    # --- Binary mask ---
    mask = torch.sigmoid(resized_logits).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)

    # --- Check if anything was detected ---
    ys, xs = np.where(binary_mask > 0)
    if ys.size == 0 or xs.size == 0:
        print("⚠️ CLIPSeg found no matching region.")
        return np.zeros((original_h, original_w), dtype=np.uint8)

    # --- SAM refinement ---
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    box = np.array([x0, y0, x1, y1])
    sam_mask = call_sam(frame, box)

    return sam_mask