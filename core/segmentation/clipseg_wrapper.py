import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class CLIPSegWrapper:
    def __init__(self, sam_wrapper, model_id):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.processor = CLIPSegProcessor.from_pretrained(model_id, backend="torchvision")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_id,
            torch_dtype=(
                torch.float16
                if self.device.type == "cuda"
                else torch.float32
            )
        ).to(self.device)
        self.model.eval()
        self.sam = sam_wrapper  # SAMWrapper instance, for refinement

    def predict(self, frame: np.ndarray,ref_image: np.ndarray | None = None, text: str | None = None) -> np.ndarray:
        if (ref_image is None) == (text is None):
            raise ValueError(
                "Provide exactly one of ref_image or text."
            )
        frame = frame.astype(np.uint8)
        image_pil = Image.fromarray(frame)
        original_h, original_w = frame.shape[:2]
        if ref_image is not None:
            ref_pil = Image.fromarray(ref_image.astype(np.uint8))
            cond = self.processor(
                images=ref_pil,
                return_tensors="pt"
            )["pixel_values"].to(self.device)

            inputs = self.processor(
                images=image_pil,
                return_tensors="pt"
            )

            inputs["conditional_pixel_values"] = cond

        else:
            inputs = self.processor(
                images=image_pil,
                text=[text.strip()],
                return_tensors="pt"
            )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            if logits.ndim == 3:
                logits = logits.unsqueeze(1)
            resized_logits = F.interpolate(
                logits, size=(original_h, original_w), mode='bilinear', align_corners=False
            )

        mask = torch.sigmoid(resized_logits).squeeze().cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)

        ys, xs = np.where(binary_mask > 0)
        if ys.size == 0 or xs.size == 0:
            return np.zeros((original_h, original_w), dtype=np.uint8)

        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
        return self.sam.predict_box(frame, box)

