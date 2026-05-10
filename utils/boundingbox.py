import cv2
import numpy as np

def get_boundary(mask, frame_resized, min_area=50):
    if mask is None or mask.size == 0:
        return None, frame_resized

    # --- Ensure mask is single-channel ---
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # --- Normalize and convert to uint8 (0 or 255) ---
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)

    # --- Morphological cleanup ---
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --- Find contours ---
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, frame_resized

    # --- Select largest contour ---
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None, frame_resized

    # --- Draw bounding box ---
    x, y, w, h = cv2.boundingRect(contour)
    frame_with_box = frame_resized.copy()
    cv2.rectangle(frame_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return (x, y, w, h), frame_with_box
