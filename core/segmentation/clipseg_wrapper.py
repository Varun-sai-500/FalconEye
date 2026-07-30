import torch
import numpy as np
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

class CLIPSegWrapper:
    def __init__(self, sam_wrapper, model_id: str):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Select target precision strategy based on hardware compatibility
        if self.device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(self.device)
            if major >= 8:
                self.dtype = torch.bfloat16
                print(f"[INFO] Compute capability {major}.{minor} >= 8.0 detected. Initializing CLIPSeg with bfloat16.")
            else:
                self.dtype = torch.float16
        elif self.device.type == "mps":
            self.dtype = torch.float16  # MPS naturally prefers float16 for mixed precision
        else:
            self.dtype = torch.float32

        self.processor = CLIPSegProcessor.from_pretrained(model_id, backend="torchvision")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self.sam = sam_wrapper

    @torch.inference_mode()
    def predict(
        self,
        frame: np.ndarray,
        *,
        ref_image: np.ndarray | None = None,
        text: str | None = None,
        top_k_points: int = 5,
        pool_kernel_size: int = 7,
        initial_bbox: tuple[int, int, int, int] | list[int] | None = None,
        rel_threshold: float = 0.7  # Only accept peaks within 70% of the max peak's value
    ) -> tuple[int, int, int, int]:

        if (ref_image is None) == (text is None):
            raise ValueError("Provide exactly one of ref_image or text.")

        original_h, original_w = frame.shape[:2]

        # Preprocessing & Inference
        if ref_image is not None:
            inputs = self.processor(images=frame, return_tensors="pt")
            cond_inputs = self.processor(images=ref_image, return_tensors="pt")
            inputs["conditional_pixel_values"] = cond_inputs["pixel_values"]
        else:
            inputs = self.processor(images=frame, text=[text.strip()], return_tensors="pt")

        inputs = {
            k: v.to(self.device, dtype=self.dtype if torch.is_floating_point(v) else None)
            for k, v in inputs.items()
        }

        # Core Segmentation Inference Pass within the targeted autocast block
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=autocast_enabled):
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # (H_low, W_low), e.g., 352x352

        h_low, w_low = logits.shape

        if initial_bbox is not None:
            # Parse [x, y, w, h] from FastAPI payload
            bx, by, bw, bh = initial_bbox

            # Map original pixel coordinates to low-res logit coordinates
            scale_x_down = w_low / original_w
            scale_y_down = h_low / original_h

            x1 = int(np.floor(bx * scale_x_down))
            y1 = int(np.floor(by * scale_y_down))
            x2 = int(np.ceil((bx + bw) * scale_x_down))
            y2 = int(np.ceil((by + bh) * scale_y_down))

            # Pad the bounding box slightly to capture context
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_low, x2 + pad)
            y2 = min(h_low, y2 + pad)

            # Create a spatial mask on the GPU and discard everything outside the ROI
            spatial_mask = torch.zeros_like(logits, dtype=torch.bool)
            spatial_mask[y1:y2, x1:x2] = True
            logits = torch.where(spatial_mask, logits, torch.full_like(logits, float("-inf")))

        # Local Peak Suppression (Upcast to float32 explicitly for safe pooling operations)
        logits_4d = logits.unsqueeze(0).unsqueeze(0).float()
        max_pooled = F.max_pool2d(
            logits_4d,
            kernel_size=pool_kernel_size,
            stride=1,
            padding=pool_kernel_size // 2,
        ).squeeze().to(dtype=self.dtype)

        local_peaks_mask = logits == max_pooled
        suppressed_logits = torch.where(
            local_peaks_mask, logits, torch.full_like(logits, float("-inf"))
        )

        max_val = torch.max(suppressed_logits)
        if torch.isinf(max_val):
            raise ValueError("CLIPSeg produced no valid local peaks.")

        # Filter out weak background noise peaks that are far lower than the max peak
        confidence_mask = suppressed_logits > (max_val - 2.0)
        suppressed_logits = torch.where(
            confidence_mask, suppressed_logits, torch.full_like(suppressed_logits, float("-inf"))
        )

        # Dynamic Top-K Selection
        flat_logits = suppressed_logits.flatten()
        valid_indices = torch.nonzero(torch.isfinite(flat_logits), as_tuple=False).squeeze(1)

        if valid_indices.numel() == 0:
            raise ValueError("CLIPSeg produced no stable local peaks within threshold.")

        valid_logits = flat_logits[valid_indices]
        k = min(top_k_points, valid_logits.numel())

        _, order = torch.topk(valid_logits, k=k)
        top_indices = valid_indices[order]

        # Coordinate Extraction & Native Scaling
        ys_low = torch.div(top_indices, w_low, rounding_mode="trunc")
        xs_low = top_indices % w_low

        scale_x = original_w / w_low
        scale_y = original_h / h_low

        xs_orig = torch.round(xs_low.float() * scale_x).long()
        ys_orig = torch.round(ys_low.float() * scale_y).long()

        xs_orig.clamp_(0, original_w - 1)
        ys_orig.clamp_(0, original_h - 1)

        point_list = torch.stack((xs_orig, ys_orig), dim=1).cpu().tolist()

        # Hand off spatially isolated points to SAM pipeline
        return self.sam.predict(frame, point_list=point_list)