from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
from PIL import Image
import torch
import cv2
import numpy as np
from sam_model import segment_on_click
from image_preprocessing import preprocess_frame
from control import RoverController
from boundingbox import get_boundary
from DaSiamRPN.dasiam_tracker import DaSiamRPNTracker
from clipseg_model import clipping

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
print("Capturing frame...")
time.sleep(2)
ret, frame = cap.read()
if not ret:
    print("cannot capture frame")
else:
    rgb_frame , frame_resized = preprocess_frame(frame)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)   
cap.release()
cv2.destroyAllWindows()

print("Choose an option:")
print("1. Click")
print("2. Reference image")
print("3. Text")

choice = input("Enter 1, 2, or 3: ").strip()
if choice=='1':
    clicks=int(input("Enter number of clicks: "))
    mask = segment_on_click(rgb_frame,clicks)         
elif choice=='2':
    Tk().withdraw()
    image_path = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    ref_image = Image.open(image_path).convert("RGB")
    ref_image = np.array(ref_image)
    ref_image = cv2.resize(ref_image, (512, 512))
    # mask = clipper(rgb_frame,ref_image=ref_image)
    mask=clipping(rgb_frame,ref_image=ref_image)
elif choice=='3':
    text_prompt = input("Enter text prompt: ")
    # mask= clipper(rgb_frame, text=text_prompt)
    mask=clipping(rgb_frame, text=text_prompt)
else:
    print("Invalid choice. Exiting.")
    exit()

bbox, frame_with_box = get_boundary(mask, frame_resized)
if bbox:
    print("Bounding box:", bbox)
    cv2.imshow("Tracked Object", frame_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rover = RoverController("./rover_controller")

tracker = DaSiamRPNTracker()
bbox = tracker.init_from_mask(rgb_frame, mask)
print("[INFO] Initialized with bbox:", bbox)
try:
    for box in tracker.track_live(video_src=0, display=True):
        print("BBox:", box)

        if box is None:
            bbox = None
        else:
            bbox = tuple(box)

        ctrl_out = rover.send_bbox(bbox)
        print("[CTRL]", ctrl_out)

except KeyboardInterrupt:
    print("🛑 Tracking stopped")

finally:
    rover.close()
