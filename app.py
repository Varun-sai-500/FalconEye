"""
FalconEye — Gradio Frontend
Pure HTTP/WS client. Zero ML imports.
Flow: Capture → Segment → Init Tracker → Track / Follow
"""

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
from queue import Queue, Empty

from core.utils.image_preprocessing import pil_to_bgr, bgr_to_pil

API_BASE = "http://localhost:8000"
WS_BASE  = "ws://localhost:8000"

state = {
    "mask_b64":       None,
    "last_frame_bgr": None,   # captured frame, used for segment + track init
    "stop_flag":      False,
    "ws_error":       None,
    "tracking":       False,
}
result_queue     = Queue(maxsize=1)
follow_queue     = Queue(maxsize=1)
live_frame_queue = Queue(maxsize=1)   # frames streamed in from the browser webcam
api_client    = httpx.Client(timeout=30)
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

def draw_bbox(
    frame_bgr,
    bbox,
    backend=None,
    model_fps=None,
    tracker_fps=None,
    score=None,
    lost=False,
):
    x, y, w, h = [int(v) for v in bbox]

    out = frame_bgr.copy()

    color = (0, 80, 255) if lost else (0, 220, 0)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    if score is not None:
        label = f"{'LOST' if lost else 'OK'}  {score:.2f}"
        cv2.putText(
            out,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    hud = []

    if backend is not None:
        hud.append(str(backend))

    if tracker_fps is not None:
        hud.append(f"Tracker: {tracker_fps:.1f} FPS")

    if model_fps is not None:
        hud.append(f"Model: {model_fps:.1f} FPS")

    if hud:
        cv2.putText(
            out,
            " | ".join(hud),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    return out


def prep_frame(rgb_np):
    """Normalize a browser-supplied RGB numpy frame into the working BGR format.
    This is the only place frame pre-processing happens now, since both the
    one-shot capture and the streaming loop funnel through here."""
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (512, 512))
    bgr = cv2.flip(bgr, 1)
    return bgr


def follow_worker():
    while True:
        bbox = follow_queue.get()

        try:
            follow_client.post(
                f"{API_BASE}/follow/command",
                json={"bbox": bbox},
            )
        except Exception as e:
            print(e)

# ── Step 1: Capture (browser webcam — no server-side VideoCapture) ──
def capture_frame(np_img):
    """np_img is whatever the browser's webcam widget currently holds
    (an RGB numpy array). We never open a camera device on the server."""
    if np_img is None:
        return None, "No webcam frame yet — allow camera access in your browser and try again."
    frame_bgr = prep_frame(np_img)
    state["last_frame_bgr"] = frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), "Frame captured. Now segment."

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
        return "Run segmentation first.", gr.update(visible=False), gr.update(visible=False)
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
        return f"Track init failed: {e}", gr.update(visible=False), gr.update(visible=False)
    bbox = resp.json()["bbox"]
    # live_group AND the streaming webcam both become visible so the browser
    # starts pushing frames once tracking is actually possible.
    return f"Tracker ready. bbox={bbox}", gr.update(visible=True), gr.update(visible=True)

# ── Step 4: Track / Follow ────────────────────────────────────
def stop_all():
    state["stop_flag"] = True
    state["tracking"]  = False
    return "Stopped.", None, ""

def push_live_frame(rgb_np):
    """Wired to the streaming webcam component's .stream() event. The browser
    calls this repeatedly with fresh frames; we just drop them in a queue for
    the background WS thread to consume. Still zero server-side camera access."""
    if rgb_np is None:
        return
    frame_bgr = prep_frame(rgb_np)
    if live_frame_queue.full():
        try:
            live_frame_queue.get_nowait()
        except Empty:
            pass
    live_frame_queue.put(frame_bgr)

def _ws_thread(mode):
    """Runs in background thread — reads frames pushed in from the browser
    webcam (via live_frame_queue) and streams them to the backend WS."""
    async def run():
        try:
            async with websockets.connect(f"{WS_BASE}/track/live") as ws:
                while not state["stop_flag"]:
                    try:
                        frame = live_frame_queue.get(timeout=1.0)
                    except Empty:
                        continue

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
                            if follow_queue.full():
                                follow_queue.get_nowait()
                            follow_queue.put(result["bbox"])
                        except Exception as e:
                            print(e)
        except Exception as e:
            state["ws_error"] = str(e)
        finally:
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

            bbox = result["bbox"]
            model_fps = result["model_fps"]
            tracker_fps = result["tracker_fps"]
            backend = result["backend"]
            score = result["score"]
            lost = result["lost"]

            vis = draw_bbox(
                frame_bgr,
                bbox,
                backend,
                model_fps,
                tracker_fps,
                score,
                lost,
            )

            status = (
                "follow mode active" if not lost else "LOST"
            ) if mode == "follow" else (
                f"{'LOST' if lost else 'tracking'}  score={score:.2f}"
            )

            fps_str = (
                f'<p class="fps-display">'
                f'Model: {model_fps:.1f} FPS | '
                f'Tracker: {tracker_fps:.1f} FPS'
                f'</p>'
            )

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

with gr.Blocks(title="FalconEye: A Modular Prompt-Guided Perception and Tracking System") as demo:

    gr.Markdown("## 🚀 FalconEye: A Modular Prompt-Guided Perception and Tracking System\nSegment · Track · Follow")

    with gr.Row(equal_height=False):

        # ── LEFT ──────────────────────────────────────────────
        with gr.Column(scale=5):

            gr.HTML('<p class="step-label">Capture frame</p>')
            gr.HTML(
                '<p class="status-txt">Allow camera access below, then press the '
                'shutter/camera icon INSIDE the widget to snap a photo.</p>'
            )
            # Webcam widget: only job is snapping a photo. Its .select() click
            # events are unreliable in Gradio while it's in webcam-source mode,
            # so we never try to mark click-points directly on this one.
            webcam_widget = gr.Image(
                label="Webcam",
                type="numpy",
                sources=["webcam"],
                streaming=False,
                interactive=True,
            )
            # Plain display/annotation image — populated from webcam_widget after
            # a snapshot. Not a webcam source itself, so .select() click events
            # fire reliably for click-based segmentation.
            input_frame = gr.Image(
                label="Captured frame (click here to mark segmentation points)",
                type="numpy",
                interactive=True,
                sources=[],
            )
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

            with gr.Column(visible=False) as action_group:
                gr.HTML('<p class="step-label" style="margin-top:10px;">3 — tracking</p>')
                track_init_btn    = gr.Button("Initialize tracker", variant="secondary")
                track_init_status = gr.HTML('<p class="status-txt"></p>')

                with gr.Column(visible=False) as live_group:
                    with gr.Row():
                        track_btn  = gr.Button("Track",  variant="primary")
                        follow_btn = gr.Button("Follow", variant="primary")
                        stop_btn   = gr.Button("Stop",   variant="stop")

                    gr.HTML(
                        '<p class="status-txt">Press the button below to start live feed, then click Track or Follow. </p>'
                    )
                    # This is the ONLY thing that keeps the browser camera alive
                    # during Track/Follow. It streams frames straight into
                    # push_live_frame() via .stream() below 
                    live_webcam = gr.Image(
                        label="Live Camera",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        height=160,
                    )

    # ── Wiring ────────────────────────────────────────────────

    # capture — fires the moment the webcam widget actually holds a snapshot
    # (i.e. right after the user presses its internal shutter icon), rather
    # than waiting for a separate button that could read a stale/empty value.
    webcam_widget.change(
        capture_frame,
        inputs=[webcam_widget],
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

    # init tracker — also reveals the streaming webcam so the browser starts
    # pushing live frames only once there's a tracker to feed them to
    track_init_btn.click(
        init_tracker, [],
        [track_init_status, live_group, live_webcam],
    )

    # live webcam stream -> queue consumed by the background WS thread
    live_webcam.stream(
        push_live_frame,
        inputs=[live_webcam],
        outputs=[],
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
