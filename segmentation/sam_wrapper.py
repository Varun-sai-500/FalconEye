import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


class SAMWrapper:
    def __init__(self, model_id):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.processor = SamProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def predict_points(
        self,
        rgb_frame: np.ndarray,
        point_list: list[list[int]],
        point_labels: list[int] | None = None,
        multimask_output: bool = False,
    ) -> np.ndarray:
        """
        rgb_frame: HxWx3 RGB uint8 numpy array (already preprocessed/resized
                   by preprocess_frame upstream).
        point_list: [[x, y], [x, y], ...] in pixel coords of rgb_frame.
        point_labels: 1 = foreground click, 0 = background click.
                      Defaults to all-foreground if not provided.

        Returns: HxW bool/uint8 mask (same H,W as rgb_frame) — single mask,
                 already picked as the best of SAM's 3 candidate masks.
        """
        if not point_list:
            raise ValueError("point_list must contain at least one point")

        if point_labels is None:
            point_labels = [1] * len(point_list)

        if len(point_labels) != len(point_list):
            raise ValueError("point_list and point_labels must be the same length")

        image = Image.fromarray(rgb_frame)

        # HF SAM expects nested lists: batch -> points-per-object -> [x, y]
        input_points = [[point_list]]
        input_labels = [[point_labels]]

        inputs = self.processor(
            image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs, multimask_output=multimask_output)

        # post_process_masks handles unpadding + resize back to original image size
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0] shape: (num_objects, num_candidate_masks, H, W)
        candidate_masks = masks[0][0]  # (num_candidate_masks, H, W)

        if multimask_output:
            # Pick highest-IoU-scoring mask
            iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
            best_idx = int(np.argmax(iou_scores))
            best_mask = candidate_masks[best_idx]
        else:
            best_mask = candidate_masks[0]

        return best_mask.numpy().astype(bool)
    @torch.inference_mode()
    def predict_box(
        self,
        rgb_frame: np.ndarray,
        box: np.ndarray,
        multimask_output: bool = False,
    ) -> np.ndarray:
        """
        rgb_frame: HxWx3 RGB uint8 numpy array.
        box: shape (4,) as [x0, y0, x1, y1] in pixel coords of rgb_frame.

        Returns: HxW bool mask (same H,W as rgb_frame).
        """
        if box.shape != (4,):
            raise ValueError("Box must have shape (4,) as [x0, y0, x1, y1].")

        image = Image.fromarray(rgb_frame)

        # HF SAM expects nested lists: batch -> boxes-per-object -> [x0, y0, x1, y1]
        input_boxes = [[box.tolist()]]

        inputs = self.processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs, multimask_output=multimask_output)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        candidate_masks = masks[0][0]  # (num_candidate_masks, H, W)

        if multimask_output:
            iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
            best_idx = int(np.argmax(iou_scores))
            best_mask = candidate_masks[best_idx]
        else:
            best_mask = candidate_masks[0]

        return best_mask.numpy().astype(bool)