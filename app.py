"""
FalconEye — Gradio Frontend
Pure HTTP/WS client. Zero ML imports.
Flow: Capture → Segment → Init Tracker → Track / Follow
"""

import sys
import gradio as gr
import httpx
import websockets
import asyncio
import cv2
import numpy as np
import base64
import json
import threading
import time
from queue import Queue

from core.utils.image_preprocessing import pil_to_bgr, bgr_to_pil

API_BASE = "http://localhost:8000"
WS_BASE  = "ws://localhost:8000"

state = {
    "mask_b64":       None,
    "last_frame_bgr": None,   # captured frame, used for segment + track init
    "stop_flag":      False,
    "ws_result":      None,
    "ws_error":       None,
    "tracking":       False,
}
result_queue = Queue(maxsize=1)
follow_queue = Queue(maxsize=1)
api_client = httpx.Client(timeout=30)
follow_client = httpx.Client(timeout=0.05)

# ── Helpers ───────────────────────────────────────────────────
def numpy_to_bytes(frame_bgr, quality=70):
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def decode_mask_b64(mask_b64):
    mask_bytes = base64.b64decode(mask_b64)
    return cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

def overlay_mask(frame_bgr, mask):
    out = frame_bgr.copy()
    green = out.copy()
    green[mask > 127] = [0, 200, 0]
    return cv2.addWeighted(green, 0.45, out, 0.55, 0)

def draw_bbox(frame_bgr, bbox, score=None, lost=False):
    x, y, w, h = [int(v) for v in bbox]
    out = frame_bgr.copy()
    color = (0, 80, 255) if lost else (0, 220, 0)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    if score is not None:
        label = f"{'LOST' if lost else 'OK'}  {score:.2f}"
        cv2.putText(out, label, (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out

def follow_worker():
    while True:
        bbox = follow_queue.get()

        try:
            follow_client.post(
                f"{API_BASE}/follow/command",
                json={"bbox": bbox},
            )
        except Exception:
            pass

# ── Step 1: Capture ───────────────────────────────────────────
def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Failed to open webcam."
    ret, frame = cap.read()
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.flip(frame, 1)
    cap.release()
    if not ret:
        return None, "Failed to capture frame."
    state["last_frame_bgr"] = frame.copy()
    # return RGB numpy for display in gr.Image
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "Frame captured. Now segment."

# ── Click collector ───────────────────────────────────────────

def on_image_click(np_img, method, click_points_state, evt: gr.SelectData):
    if method != "click":
        return np_img, json.dumps(click_points_state), click_points_state

    x, y = evt.index
    click_points_state = click_points_state + [[x, y]]

    clean = state["last_frame_bgr"]
    clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    vis = clean_rgb.copy()
    for px, py in click_points_state:
        cv2.circle(vis, (px, py), 7, (255, 60, 60), -1)
    return vis, json.dumps(click_points_state), click_points_state

def clear_clicks(np_img):
    return np_img, "[]", []

# ── Step 2: Segment ───────────────────────────────────────────
def run_segment(np_img, method, click_pts_json, text_prompt, ref_pil):
    if np_img is None:
        return None, None, "Capture a frame first.", gr.update(visible=False)
    frame_bgr = state["last_frame_bgr"]
    state["last_frame_bgr"] = frame_bgr

    files = {"file": ("frame.jpg", numpy_to_bytes(frame_bgr), "image/jpeg")}
    data  = {"method": method}

    if method == "click":
        pts = json.loads(click_pts_json or "[]")
        if not pts:
            return None, None, "No click points recorded.", gr.update(visible=False)
        data["points"] = json.dumps(pts)
    elif method == "text":
        if not text_prompt.strip():
            return None, None, "Enter a text prompt.", gr.update(visible=False)
        data["text"] = text_prompt.strip()
    elif method == "reference":
        if ref_pil is None:
            return None, None, "Upload a reference image.", gr.update(visible=False)
        files["ref_file"] = ("ref.jpg", numpy_to_bytes(pil_to_bgr(ref_pil)), "image/jpeg")

    try:
        resp =  api_client.post(
            f"{API_BASE}/segment",
            files=files,
            data=data,
        )
        resp.raise_for_status()
    except Exception as e:
        return None, None, f"Segment failed: {e}", gr.update(visible=False)

    result   = resp.json()
    mask_b64 = result["mask_b64"]
    bbox     = result.get("bbox")
    state["mask_b64"] = mask_b64

    mask = decode_mask_b64(mask_b64)
    vis  = overlay_mask(frame_bgr, mask)
    if bbox:
        vis = draw_bbox(vis, bbox)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"Done. bbox={bbox}", gr.update(visible=True)
# ── Step 3: Init tracker ──────────────────────────────────────
def init_tracker():
    if state["last_frame_bgr"] is None or state["mask_b64"] is None:
        return "Run segmentation first.", gr.update(visible=False)
    files = {"file": ("frame.jpg", numpy_to_bytes(state["last_frame_bgr"]), "image/jpeg")}
    data  = {"mask_b64": state["mask_b64"]}
    try:
        resp = api_client.post(
            f"{API_BASE}/track/init",
            files=files,
            data=data,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Track init failed: {e}", gr.update(visible=False)
    bbox = resp.json()["bbox"]
    return f"Tracker ready. bbox={bbox}", gr.update(visible=True)

# ── Step 4: Track / Follow ────────────────────────────────────
def stop_all():
    state["stop_flag"] = True
    state["tracking"]  = False
    return "Stopped.", None, ""

def _ws_thread(mode):
    """Runs in background thread — opens webcam, streams to WS."""
    async def run():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            state["ws_error"] = "Cannot open webcam."
            state["tracking"] = False
            return
        try:
            async with websockets.connect(f"{WS_BASE}/track/live") as ws:
                while not state["stop_flag"]:
                    ret, frame = cap.read()
                    frame = cv2.resize(frame, (512, 512))
                    frame = cv2.flip(frame, 1)
                    jpg = numpy_to_bytes(frame)
                    await ws.send(jpg)
                    raw = await ws.recv()
                    result = json.loads(raw)

                    if "error" in result:
                        state["ws_error"] = result["error"]
                        break
                    if result_queue.full():
                        result_queue.get_nowait()

                    result_queue.put((frame, result))
                    state["ws_error"] = None

                    if mode == "follow" and not result.get("lost"):
                        try:
                            if mode == "follow" and not result.get("lost"):
                                if follow_queue.full():
                                    follow_queue.get_nowait()
                                follow_queue.put(result["bbox"])
                        except Exception:
                            pass
        except Exception as e:
            state["ws_error"] = str(e)
        finally:
            cap.release()
            state["tracking"] = False

    asyncio.run(run())

def _start_and_poll(mode):
    state["stop_flag"]  = False
    state["tracking"]   = True

    t = threading.Thread(target=_ws_thread, args=(mode,), daemon=True)
    t.start()

    while state["tracking"] or not result_queue.empty():
        if state["ws_error"]:
            yield None, f"Error: {state['ws_error']}", ""
            state["ws_error"] = None
            break
        if not result_queue.empty():
            frame_bgr, result = result_queue.get()

            bbox  = result["bbox"]
            score = result["score"]
            lost  = result["lost"]
            fps   = result["fps"]

            vis    = draw_bbox(frame_bgr, bbox, score, lost)
            status = ("follow mode active" if not lost else "LOST") if mode == "follow" \
                     else f"{'LOST' if lost else 'tracking'}  score={score:.2f}"
            fps_str = f'<p class="fps-display">{fps:.1f} fps</p>'

            yield cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), status, fps_str
        else:
            time.sleep(0.005)  # only sleep when idle, not every iteration

def do_track():
    yield from _start_and_poll("track")

def do_follow():
    yield from _start_and_poll("follow")

# ── UI ────────────────────────────────────────────────────────
css = """
.step-label {
    font-size: 11px; font-weight: 500; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--color-text-secondary);
    margin: 0 0 6px 0;
}
.status-txt { font-size: 13px; color: var(--color-text-secondary); min-height: 18px; }
.fps-display { font-size: 32px; font-weight: 600; text-align: center; padding: 6px 0; }
"""

with gr.Blocks(title="FalconEye") as demo:

    gr.Markdown("## FalconEye\nSegment · Track · Follow")

    with gr.Row(equal_height=False):

        # ── LEFT ──────────────────────────────────────────────
        with gr.Column(scale=5):

            gr.HTML('<p class="step-label">Capture frame</p>')
            input_frame = gr.Image(
                label="Captured frame",
                type="numpy",
                interactive=True, # needed for click segmentation
                sources=["webcam"]
            )
            capture_btn    = gr.Button("Capture from webcam", variant="secondary")
            capture_status = gr.HTML('<p class="status-txt"></p>')

            gr.HTML('<p class="step-label" style="margin-top:14px;">2 — segmentation method</p>')
            method = gr.Radio(
                ["click", "text", "reference"],
                value="text", label="", container=False,
            )
            text_prompt = gr.Textbox(
                label="Text prompt",
                placeholder="e.g. red car, person in blue jacket",
                visible=True,
            )
            ref_image = gr.Image(
                label="Reference image", type="pil",
                sources=["upload"], visible=False, height=150,
            )
            with gr.Row(visible=False) as click_row:
                click_pts = gr.Textbox(value="[]", visible=False)
                clear_btn = gr.Button("Clear clicks", size="sm")
            click_hint = gr.HTML(
                '<p class="status-txt">Click on the frame above to mark the object.</p>',
                visible=False,
            )

            seg_btn    = gr.Button("Segment", variant="primary", size="lg")
            seg_status = gr.HTML('<p class="status-txt"></p>')

        # ── RIGHT ─────────────────────────────────────────────
        with gr.Column(scale=5):

            gr.HTML('<p class="step-label">Output</p>')
            output_frame = gr.Image(
                label="", type="pil",
                height=320, interactive=False,
            )

            fps_display = gr.HTML('<p class="fps-display"></p>')
            track_status = gr.HTML('<p class="status-txt"></p>')

            with gr.Group(visible=False) as action_group:
                gr.HTML('<p class="step-label" style="margin-top:10px;">3 — tracking</p>')
                track_init_btn    = gr.Button("Initialize tracker", variant="secondary")
                track_init_status = gr.HTML('<p class="status-txt"></p>')

                with gr.Group(visible=False) as live_group:
                    with gr.Row():
                        track_btn  = gr.Button("Track",  variant="primary")
                        follow_btn = gr.Button("Follow", variant="primary")
                        stop_btn   = gr.Button("Stop",   variant="stop")

    # ── Wiring ────────────────────────────────────────────────

    # capture
    capture_btn.click(
        capture_frame,
        outputs=[input_frame, capture_status],
    )

    # method show/hide
    def update_method(m):
        return (
            gr.update(visible=(m == "text")),
            gr.update(visible=(m == "reference")),
            gr.update(visible=(m == "click")),
            gr.update(visible=(m == "click")),
        )
    method.change(update_method, method,
                  [text_prompt, ref_image, click_row, click_hint])
    click_points_state = gr.State([])
    # clicks on captured frame
    input_frame.select(
        on_image_click, [input_frame, method, click_points_state],
        [input_frame, click_pts, click_points_state],
    )
    clear_btn.click(clear_clicks, [input_frame], [input_frame, click_pts, click_points_state])

    # segment
    seg_btn.click(
        run_segment,
        [input_frame, method, click_pts, text_prompt, ref_image],
        [output_frame, seg_status, action_group],
    )

    # init tracker
    track_init_btn.click(
        init_tracker, [],
        [track_init_status, live_group],
    )

    # track + follow — streaming=True is critical
    track_btn.click(
        do_track, [],
        [output_frame, track_status, fps_display],
    )
    follow_btn.click(
        do_follow, [],
        [output_frame, track_status, fps_display],
    )

    # stop
    stop_btn.click(
        stop_all, [],
        [track_status, output_frame, fps_display],
    )

if __name__ == "__main__":
    threading.Thread(
        target=follow_worker,
        daemon=True,
    ).start()
    demo.launch(css=css, theme=gr.themes.Soft(), server_port=7860)