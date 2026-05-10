import cv2

# --- Preprocess frame for SAM ---
def preprocess_frame(frame, target_size=(512, 512)):
    frame_resized = cv2.resize(frame, target_size)
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return rgb_frame, frame_resized

def get_bgr(frame_rgb):
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)