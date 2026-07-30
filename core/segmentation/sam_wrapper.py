import torch
import numpy as np
from transformers import SamModel, SamProcessor

class SAMWrapper:
    def __init__(self, model_id: str):
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
                print(f"[INFO] Compute capability {major}.{minor} >= 8.0 detected. Initializing SAM with bfloat16.")
            else:
                self.dtype = torch.float16
                print(f"[INFO] Compute capability {major}.{minor} < 8.0 detected. Initializing SAM with float16.")
        elif self.device.type == "mps":
            self.dtype = torch.float16  # MPS naturally prefers float16 for mixed precision
            print("[INFO] MPS device detected. Initializing SAM with float16.")
        else:
            self.dtype = torch.float32

        self.model = SamModel.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.processor = SamProcessor.from_pretrained(model_id)

    def _mask_to_bbox(self, mask: torch.Tensor) -> tuple[int, int, int, int]:
        mask = mask > 0.5
        coords = torch.nonzero(mask)

        if coords.numel() == 0:
            raise ValueError("SAM produced an empty mask.")

        ys = coords[:, 0]
        xs = coords[:, 1]

        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()

        bbox = torch.stack((
            x1,
            y1,
            x2 - x1 + 1,
            y2 - y1 + 1,
        )).cpu().tolist()

        return tuple(map(int, bbox))

    @torch.inference_mode()
    def predict(
        self,
        rgb_frame: np.ndarray,
        *,
        point_list: list[list[int]] | None = None,
        multimask_output: bool = False,
    ) -> tuple[int, int, int, int]:
        if not point_list:
            raise ValueError("point_list must contain at least one point")

        # Format positive prompt labels (1 means foreground point) for all top-K coordinates
        point_labels = [1] * len(point_list)

        # Build batched inputs matching SamProcessor's required dimensions [[[X, Y]]]
        inputs = self.processor(
            images=rgb_frame,
            input_points=[[point_list]],
            input_labels=[[point_labels]],
            return_tensors="pt",
        )

        # Enforce target dtype precision bounds directly during model transfer
        model_inputs = {
            "pixel_values": inputs["pixel_values"].to(self.device, dtype=self.dtype),
            "input_points": inputs["input_points"].to(self.device),  # Coordinates remain positional long/float
            "input_labels": inputs["input_labels"].to(self.device),
            "multimask_output": multimask_output,
        }

        original_sizes = inputs["original_sizes"]
        reshaped_input_sizes = inputs["reshaped_input_sizes"]

        # Core SAM Inference Pass executing inside targeted autocast block
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=autocast_enabled):
            outputs = self.model(**model_inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks,
            original_sizes,
            reshaped_input_sizes,
        )

        candidate_masks = masks[0][0]

        if multimask_output:
            best_mask = candidate_masks[torch.argmax(outputs.iou_scores[0, 0])]
        else:
            best_mask = candidate_masks[0]

        # Extract coordinates directly from SAM's final mask output
        return self._mask_to_bbox(best_mask)