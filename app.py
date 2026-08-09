import sys
import json
import time
import cv2
import threading
import shutil
from enum import Enum, auto
from pathlib import Path
from fractions import Fraction
import numpy as np
import av

from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl, QByteArray, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply, QHttpMultiPart, QHttpPart
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QFileDialog, QLineEdit, QStackedWidget, QMessageBox
)

API_SEGMENT = "http://127.0.0.1:8000/segment"
API_FOLLOW  = "http://127.0.0.1:8000/follow"
WS_TRACK    = "ws://127.0.0.1:8000/track/live"


# ----------------------------------------------------------------------
# Source mode enum — replaces fragile "webcam"/"video" string literals
# ----------------------------------------------------------------------

class SourceMode(Enum):
    WEBCAM = auto()
    VIDEO  = auto()


# ----------------------------------------------------------------------
# Real-Time H.264 Encoder Helper using PyAV
# ----------------------------------------------------------------------

class H264Encoder:
    def __init__(self, width=640, height=480, fps=30, bitrate=1_000_000):
        self.width  = width
        self.height = height

        self.codec = av.CodecContext.create('h264', 'w')
        self.codec.width     = width
        self.codec.height    = height
        self.codec.pix_fmt   = 'yuv420p'
        self.codec.time_base = Fraction(1, fps)
        self.codec.bit_rate  = bitrate
        self.codec.gop_size  = fps  # One keyframe per second
        self.codec.options   = {
            'preset':         'ultrafast',
            'tune':           'zerolatency',
            'repeat-headers': '1',  # SPS/PPS in every keyframe — decoder-safe
        }
        self.codec.open()

        # Pre-allocated YUV420P frame — avoids a new heap allocation per encode.
        self.yuv_frame = av.VideoFrame(width, height, format='yuv420p')

        # Pre-allocated I420 scratch buffer for OpenCV's colorspace conversion.
        self.i420_buffer = np.empty((height * 3 // 2, width), dtype=np.uint8)

    def encode(self, cv_bgr_frame: np.ndarray) -> bytes:
        # BGR -> I420 directly via OpenCV — skips libswscale entirely.
        cv2.cvtColor(cv_bgr_frame, cv2.COLOR_BGR2YUV_I420, dst=self.i420_buffer)

        h, w = self.height, self.width

        # I420 layout produced by OpenCV:
        #   Y:  rows [0       : h      ]  — h   × w
        #   U:  rows [h       : h+h//4 ]  — h/4 × w   (packed: h/4 rows of width w)
        #   V:  rows [h+h//4  : h+h//2 ]  — h/4 × w
        #
        # Reshape U and V from OpenCV's packed layout (h//4, w) to the geometric plane layout (h//2, w//2)
        y = self.i420_buffer[:h,            :]
        u = self.i420_buffer[h:h+h//4,      :].reshape(h // 2, w // 2)
        v = self.i420_buffer[h+h//4:h+h//2, :].reshape(h // 2, w // 2)

        # Write directly into PyAV's pre-allocated plane buffers.
        # Renamed (h, w) to (ph, pw) inside the loop to avoid variable shadowing!
        for plane, data, (ph, pw) in (
            (self.yuv_frame.planes[0], y, (h,      w     )),
            (self.yuv_frame.planes[1], u, (h // 2, w // 2)),
            (self.yuv_frame.planes[2], v, (h // 2, w // 2)),
        ):
            arr = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
            arr[:ph, :pw] = data  # Strict assignment preventing stride overwrites

        packets = self.codec.encode(self.yuv_frame)
        payload = bytearray()
        for packet in packets:
            payload.extend(bytes(packet))
        return bytes(payload)
# ----------------------------------------------------------------------
# Encoder Worker — runs encode() on its own thread, not the GUI thread
# ----------------------------------------------------------------------

class EncoderWorker(QThread):
    encoded = Signal(bytes)

    def __init__(self, width: int, height: int, fps: int = 30):
        super().__init__()
        self.encoder      = H264Encoder(width, height, fps)
        self._frame       = None
        self._lock        = threading.Lock()
        self._frame_ready = threading.Event()
        self._running     = True

    def submit_frame(self, frame: np.ndarray):
        new_frame = frame.copy()
        with self._lock:
            self._frame = new_frame
        self._frame_ready.set()

    def stop(self):
        self._running = False
        self._frame_ready.set()  # Wake the thread if it is sleeping.
        self.wait()

    def run(self):
        while self._running:
            self._frame_ready.wait()

            if not self._running:
                break

            # Drain all submitted frames before sleeping again.
            while self._running:
                with self._lock:
                    if self._frame is None:
                        # No pending work — clear the event while still
                        # holding the lock so submit_frame() cannot race.
                        self._frame_ready.clear()
                        break
                    frame       = self._frame
                    self._frame = None

                h264_bytes = self.encoder.encode(frame)
                if h264_bytes:
                    self.encoded.emit(h264_bytes)


# ----------------------------------------------------------------------
# Uploaded Video Playback Worker
# ----------------------------------------------------------------------

class VideoWorker(QThread):
    frame_received = Signal(np.ndarray, float)

    def __init__(self, video_path: str, start_frame: int = 1):
        super().__init__()
        self.video_path  = video_path
        self.start_frame = start_frame
        self._running    = True

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        # Skip the frame(s) already shown in the frozen preview.
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        frame_interval   = 1.0 / fps
        next_frame_time  = time.perf_counter()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            now     = time.perf_counter()
            latency = max(0.0, (now - next_frame_time) * 1000.0)
            self.frame_received.emit(frame, latency)

            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Fell behind — reset rather than accumulating debt.
                next_frame_time = time.perf_counter()

        cap.release()


# ----------------------------------------------------------------------
# Interactive Video Canvas
# Overlays (bbox, click points) are painted via QPainter on every
# update_display() call, always starting from the clean base_pixmap so
# we never burn overlays into the source image.
# ----------------------------------------------------------------------

class VideoCanvas(QLabel):
    point_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.click_enabled = True
        self.setStyleSheet("background-color: #121212; border: 1px solid #2a2a2a;")

        self.current_frame   = None
        self.base_pixmap     = None  # Clean frame, no overlays
        self.display_pixmap  = None  # Painted frame shown on screen
        self.points          = []
        self.last_bbox       = None

    def set_frame(self, cv_img: np.ndarray):
        self.current_frame = cv_img

        rgb   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w  = rgb.shape[:2]
        q_img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

        # base_pixmap is always overlay-free — only updated on new frames.
        self.base_pixmap = QPixmap.fromImage(q_img)
        self.update_display()

    def set_bbox(self, bbox):
        self.last_bbox = bbox
        self.update_display()

    def clear_overlays(self):
        self.points.clear()
        self.last_bbox = None
        self.update_display()

    def mousePressEvent(self, event):
        if not self.click_enabled:
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.current_frame is not None
            and self.base_pixmap is not None
        ):
            pixmap = self.pixmap()
            if not pixmap or pixmap.isNull():
                return

            p_w, p_h = pixmap.width(), pixmap.height()
            l_w, l_h = self.width(), self.height()
            x_off    = (l_w - p_w) / 2
            y_off    = (l_h - p_h) / 2

            cx = event.position().x() - x_off
            cy = event.position().y() - y_off

            if 0 <= cx <= p_w and 0 <= cy <= p_h:
                fh, fw = self.current_frame.shape[:2]
                ix = int((cx / p_w) * fw)
                iy = int((cy / p_h) * fh)
                self.points.append((ix, iy))
                self.point_clicked.emit(ix, iy)
                self.update_display()

    def update_display(self):
        if self.base_pixmap is None:
            return

        # Always start from the clean base — never accumulate overlays.
        self.display_pixmap = self.base_pixmap.copy()

        painter = QPainter(self.display_pixmap)

        if self.last_bbox and len(self.last_bbox) == 4:
            x, y, w, h = self.last_bbox
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            painter.drawRect(int(x), int(y), int(w), int(h))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        for px, py in self.points:
            painter.drawEllipse(int(px - 4), int(py - 4), 8, 8)

        painter.end()

        scaled = self.display_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


# ----------------------------------------------------------------------
# Camera Worker — runs cv2.VideoCapture on its own thread
# ----------------------------------------------------------------------

class CameraWorker(QThread):
    frame_received = Signal(np.ndarray, float)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id  = camera_id
        self._running   = True
        # Written only from the camera thread; read once on WS connect.
        # int assignment is atomic under the GIL — safe without a lock.
        self.camera_fps = 0

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return

        fps_count = 0
        fps_start = time.time()

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.msleep(1)
                continue

            frame = cv2.flip(frame, 1)

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.camera_fps = int(round(fps_count / elapsed))
                fps_count       = 0
                fps_start       = time.time()

            t_capture = (time.time() - fps_start) * 1000
            self.frame_received.emit(frame, t_capture)
            self.msleep(1)

        cap.release()


# ----------------------------------------------------------------------
# Main Application Window
# ----------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Dashboard")
        self.resize(1100, 650)

        # ── Source state ──────────────────────────────────────────────
        self.source_mode          = SourceMode.WEBCAM
        self.uploaded_video_path  = None   # Resolved absolute path string
        self.uploaded_video_fps   = 0
        self.video_worker         = None
        # Flag: True while we are intentionally tearing down video_worker
        # so that on_video_finished() ignores the spurious finished signal.
        self._stopping_video      = False

        # ── Tracking state ────────────────────────────────────────────
        self.latest_raw_frame = None
        self.ref_image_bytes  = None
        self.is_frozen        = False
        self.is_tracking      = False
        self.encoder_worker   = None
        self.ws_max_buffer    = 2 * 1024 * 1024  # 2 MB back-pressure limit

        # ── Persistence ───────────────────────────────────────────────
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # ── Networking ────────────────────────────────────────────────
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_http_response)

        self.ws_client = QWebSocket()
        self.ws_client.connected.connect(self.on_ws_connected)
        self.ws_client.disconnected.connect(self.on_ws_disconnected)
        self.ws_client.textMessageReceived.connect(self.on_ws_message_received)
        self.ws_client.errorOccurred.connect(self.on_ws_error)

        self.init_ui()

        # ── Camera ────────────────────────────────────────────────────
        self.camera_thread = CameraWorker()
        self.camera_thread.frame_received.connect(self.on_webcam_frame)
        self.camera_thread.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ── Left column ───────────────────────────────────────────────
        left = QVBoxLayout()

        self.video_canvas = VideoCanvas()
        self.video_canvas.point_clicked.connect(self.on_canvas_point_clicked)
        left.addWidget(self.video_canvas, stretch=4)

        controls = QGroupBox("Control Panel")
        ctrl_layout = QVBoxLayout(controls)

        # Source selector
        source_box    = QGroupBox("Video Source")
        source_layout = QHBoxLayout(source_box)

        self.radio_webcam = QRadioButton("Webcam")
        self.radio_video  = QRadioButton("Video Upload")
        self.radio_webcam.setChecked(True)

        self.source_btn_group = QButtonGroup(self)
        self.source_btn_group.addButton(self.radio_webcam, 0)
        self.source_btn_group.addButton(self.radio_video,  1)

        # "Choose Video..." is a separate button — switching the radio
        # does NOT force-open a file dialog; the user clicks this when ready.
        self.btn_upload_video = QPushButton("Choose Video…")
        self.btn_upload_video.setEnabled(False)
        self.btn_upload_video.clicked.connect(self.select_video)

        source_layout.addWidget(self.radio_webcam)
        source_layout.addWidget(self.radio_video)
        source_layout.addWidget(self.btn_upload_video)
        source_layout.addStretch()

        ctrl_layout.addWidget(source_box)

        # Capture button (webcam only)
        self.btn_capture = QPushButton("Capture Frame")
        self.btn_capture.setStyleSheet(
            "font-weight: bold; background-color: #2196F3; color: white; padding: 8px;"
        )
        self.btn_capture.clicked.connect(self.capture_frame)
        ctrl_layout.addWidget(self.btn_capture)

        # Segmentation options
        seg_box    = QGroupBox("Segment Options")
        seg_layout = QVBoxLayout(seg_box)

        radio_row = QHBoxLayout()
        self.seg_btn_group = QButtonGroup(self)
        self.radio_click   = QRadioButton("Click")
        self.radio_ref     = QRadioButton("Reference")
        self.radio_text    = QRadioButton("Text")
        self.radio_click.setChecked(True)
        self.seg_btn_group.addButton(self.radio_click, 0)
        self.seg_btn_group.addButton(self.radio_ref,   1)
        self.seg_btn_group.addButton(self.radio_text,  2)
        radio_row.addWidget(self.radio_click)
        radio_row.addWidget(self.radio_ref)
        radio_row.addWidget(self.radio_text)
        radio_row.addStretch()
        seg_layout.addLayout(radio_row)

        self.input_stack = QStackedWidget()
        self.input_stack.addWidget(
            QLabel("Click on the video stream above to select target points.")
        )

        page_ref = QWidget()
        ref_row  = QHBoxLayout(page_ref)
        ref_row.setContentsMargins(0, 0, 0, 0)
        self.btn_upload_ref = QPushButton("Upload Reference Image…")
        self.lbl_ref_path   = QLabel("No file selected.")
        self.btn_upload_ref.clicked.connect(self.select_reference_image)
        ref_row.addWidget(self.btn_upload_ref)
        ref_row.addWidget(self.lbl_ref_path)
        ref_row.addStretch()
        self.input_stack.addWidget(page_ref)

        page_text = QWidget()
        txt_row   = QHBoxLayout(page_text)
        txt_row.setContentsMargins(0, 0, 0, 0)
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText("Enter target text prompt…")
        txt_row.addWidget(self.txt_prompt)
        self.input_stack.addWidget(page_text)

        seg_layout.addWidget(self.input_stack)

        self.btn_run_segment = QPushButton("Segment Target")
        self.btn_run_segment.clicked.connect(self.trigger_segmentation)
        seg_layout.addWidget(self.btn_run_segment)

        ctrl_layout.addWidget(seg_box)

        act_row = QHBoxLayout()
        self.btn_track  = QPushButton("Track")
        self.btn_follow = QPushButton("Follow")
        self.btn_clear  = QPushButton("Clear")
        self.btn_track.clicked.connect(self.toggle_tracking)
        self.btn_follow.clicked.connect(self.trigger_follow)
        self.btn_clear.clicked.connect(self.clear_all)
        act_row.addWidget(self.btn_track)
        act_row.addWidget(self.btn_follow)
        act_row.addWidget(self.btn_clear)
        ctrl_layout.addLayout(act_row)

        left.addWidget(controls, stretch=1)
        root.addLayout(left, stretch=3)

        # ── Right column — metrics ────────────────────────────────────
        metrics_box = QGroupBox("Backend Metrics")
        metrics_box.setMinimumWidth(300)
        self.metrics_container = QVBoxLayout(metrics_box)
        self.metrics_grid      = QGridLayout()
        self.metrics_grid.setSpacing(10)
        self.metrics_container.addLayout(self.metrics_grid)
        self.metrics_container.addStretch()
        root.addWidget(metrics_box, stretch=1)

        # Wire source-mode toggle AFTER all widgets exist
        self.source_btn_group.idClicked.connect(self.on_source_changed)
        self.seg_btn_group.idClicked.connect(self.on_seg_mode_changed)

        self.update_metrics_display({"Status": "Idle"})

    # ------------------------------------------------------------------
    # Metrics panel
    # ------------------------------------------------------------------

    def update_metrics_display(self, metrics: dict):
        while self.metrics_grid.count():
            item = self.metrics_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for row, (key, val) in enumerate(metrics.items()):
            self.metrics_grid.addWidget(QLabel(f"<b>{key}:</b>"), row, 0)
            v = QLabel(str(val))
            v.setWordWrap(True)
            self.metrics_grid.addWidget(v, row, 1)

    # ------------------------------------------------------------------
    # Source switching
    # ------------------------------------------------------------------

    def on_source_changed(self, idx: int):
        if idx == 0:
            self.switch_to_webcam()
        else:
            self.switch_to_video_mode()

    def switch_to_webcam(self):
        # No-op if already in webcam mode.
        if self.source_mode == SourceMode.WEBCAM:
            return

        self.close_websocket()
        self.stop_video_worker()

        self.source_mode         = SourceMode.WEBCAM
        self.uploaded_video_path = None
        self.uploaded_video_fps  = 0
        self.is_frozen           = False
        self.latest_raw_frame    = None

        self.video_canvas.current_frame  = None
        self.video_canvas.base_pixmap    = None
        self.video_canvas.display_pixmap = None
        self.video_canvas.clear_overlays()

        self.btn_upload_video.setEnabled(False)
        self.btn_capture.setEnabled(True)

        self.update_metrics_display({"Status": "Webcam"})

    def switch_to_video_mode(self):
        """Switch the UI into video-upload mode.

        Crucially this does NOT open a file dialog — that is left to
        the user clicking 'Choose Video…'.  This avoids the bug where
        toggling the radio forces an unwanted dialog every time.
        """
        if self.source_mode == SourceMode.VIDEO:
            return

        self.close_websocket()
        self.stop_video_worker()

        self.source_mode = SourceMode.VIDEO
        self.btn_upload_video.setEnabled(True)
        self.btn_capture.setEnabled(False)

        self.update_metrics_display({"Status": "Video mode — choose a file"})

    def select_video(self):
        """Open a file dialog and load the chosen video."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if not path:
            # User cancelled — if no video was previously loaded, revert to webcam.
            if self.uploaded_video_path is None:
                self.radio_webcam.setChecked(True)
                self.switch_to_webcam()
            return

        self.load_video(path)

    def load_video(self, path: str):
        self.close_websocket()
        self.stop_video_worker()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Video Error", "Could not open the selected video.")
            return

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            QMessageBox.critical(self, "Video Error", "Could not read the first frame.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        cap.release()

        self.source_mode         = SourceMode.VIDEO
        self.uploaded_video_path = str(Path(path).resolve())
        self.uploaded_video_fps  = max(1, int(round(fps)))

        # Save a copy to results/ — done via QTimer so the dialog closes first.
        src = Path(path)
        dst = self.results_dir / src.name
        if src.resolve() != dst.resolve():
            QTimer.singleShot(0, lambda s=src, d=dst: self._copy_video_to_results(s, d))

        # Freeze on frame 0; VideoWorker will start at frame 1 when tracking begins.
        self.is_frozen        = True
        self.is_tracking      = False
        self.latest_raw_frame = first_frame

        self.video_canvas.clear_overlays()
        self.video_canvas.set_frame(first_frame)

        self.btn_capture.setEnabled(False)

        self.update_metrics_display({
            "Status": "Video Loaded",
            "File":   src.name,
            "FPS":    self.uploaded_video_fps,
        })

    def _copy_video_to_results(self, src: Path, dst: Path):
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            QMessageBox.warning(
                self, "Save Warning",
                f"Video loaded but could not be copied to results/:\n{exc}"
            )

    def stop_video_worker(self):
        if self.video_worker is not None:
            self._stopping_video = True
            self.video_worker.stop()
            self.video_worker    = None
            self._stopping_video = False

    # ------------------------------------------------------------------
    # Frame routing — each source has its own slot so signal/mode
    # mismatches are caught at the routing layer, not buried in logic.
    # ------------------------------------------------------------------

    @Slot(np.ndarray, float)
    def on_webcam_frame(self, frame: np.ndarray, latency: float):
        if self.source_mode != SourceMode.WEBCAM:
            return
        self._handle_frame(frame, latency)

    @Slot(np.ndarray, float)
    def on_video_frame(self, frame: np.ndarray, latency: float):
        if self.source_mode != SourceMode.VIDEO:
            return
        self._handle_frame(frame, latency)

    def _handle_frame(self, frame: np.ndarray, latency: float):
        self.latest_raw_frame = frame
        if not self.is_frozen:
            self.video_canvas.set_frame(frame)
        if self.is_tracking and self.encoder_worker:
            self.encoder_worker.submit_frame(frame)

    # ------------------------------------------------------------------
    # Capture (webcam only)
    # ------------------------------------------------------------------

    def capture_frame(self):
        self.is_frozen = True
        self.video_canvas.points.clear()
        self.video_canvas.last_bbox = None
        self.video_canvas.update_display()
        self.update_metrics_display({"Status": "Frame Captured"})

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def trigger_segmentation(self):
        if self.latest_raw_frame is None:
            return

        method_id  = self.seg_btn_group.checkedId()
        method_map = {0: "click", 1: "reference", 2: "text"}
        method     = method_map[method_id]

        multi_part = QHttpMultiPart(QHttpMultiPart.ContentType.FormDataType)

        def _text_part(name: str, value: str):
            part = QHttpPart()
            part.setHeader(
                QNetworkRequest.KnownHeaders.ContentDispositionHeader,
                f'form-data; name="{name}"'
            )
            part.setBody(value.encode())
            return part

        multi_part.append(_text_part("method", method))

        _, buf = cv2.imencode('.jpg', self.latest_raw_frame)
        img_part = QHttpPart()
        img_part.setHeader(
            QNetworkRequest.KnownHeaders.ContentDispositionHeader,
            'form-data; name="file"; filename="frame.jpg"'
        )
        img_part.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "image/jpeg"
        )
        img_part.setBody(buf.tobytes())
        multi_part.append(img_part)

        if method == "click":
            multi_part.append(
                _text_part("points", json.dumps(self.video_canvas.points))
            )
        elif method == "reference" and self.ref_image_bytes:
            ref_part = QHttpPart()
            ref_part.setHeader(
                QNetworkRequest.KnownHeaders.ContentDispositionHeader,
                'form-data; name="ref_file"; filename="ref.jpg"'
            )
            ref_part.setHeader(
                QNetworkRequest.KnownHeaders.ContentTypeHeader, "image/jpeg"
            )
            ref_part.setBody(self.ref_image_bytes)
            multi_part.append(ref_part)
        elif method == "text":
            multi_part.append(
                _text_part("text", self.txt_prompt.text().strip())
            )

        request = QNetworkRequest(QUrl(API_SEGMENT))
        reply   = self.network_manager.post(request, multi_part)
        reply.setProperty("req_type", "segment")
        multi_part.setParent(reply)

    def on_seg_mode_changed(self, idx: int):
        self.input_stack.setCurrentIndex(idx)
        self.video_canvas.click_enabled = (idx == 0)
        if idx != 0:
            self.video_canvas.points.clear()
            self.video_canvas.update_display()

    # ------------------------------------------------------------------
    # Tracking — WebSocket lifecycle
    # ------------------------------------------------------------------

    def toggle_tracking(self):
        if not self.is_tracking:
            self.ws_client.open(QUrl(WS_TRACK))
        else:
            self.close_websocket()

    @Slot()
    def on_ws_connected(self):
        bbox = self.video_canvas.last_bbox
        if bbox is None or self.latest_raw_frame is None:
            self.ws_client.close()
            return

        self.video_canvas.points.clear()
        self.video_canvas.update_display()

        h, w = self.latest_raw_frame.shape[:2]
        fps  = (
            max(1, self.uploaded_video_fps)
            if self.source_mode == SourceMode.VIDEO
            else max(1, self.camera_thread.camera_fps or 30)
        )

        self.encoder_worker = EncoderWorker(width=w, height=h, fps=fps)
        self.encoder_worker.encoded.connect(self.send_encoded_frame)
        self.encoder_worker.start()

        # Send bbox metadata first, then the initial keyframe.
        self.ws_client.sendTextMessage(json.dumps({"bbox": bbox}))
        self.encoder_worker.submit_frame(self.latest_raw_frame)

        # is_tracking is set only after backend confirms initialisation.

    @Slot(str)
    def on_ws_message_received(self, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("status") == "initialized":
            self.is_tracking = True
            self.is_frozen   = False
            self.btn_track.setText("Stop Track")

            if self.source_mode == SourceMode.VIDEO:
                # Start streaming the video from frame 1 (frame 0 was the
                # frozen segmentation preview sent at WS connect time).
                self.stop_video_worker()
                self.video_worker = VideoWorker(
                    self.uploaded_video_path,
                    start_frame=1,
                )
                self.video_worker.frame_received.connect(self.on_video_frame)
                self.video_worker.finished.connect(self.on_video_finished)
                self.video_worker.start()
            return

        if "bbox" in data:
            self.video_canvas.set_bbox(data["bbox"])
        self.update_metrics_display(data)

    @Slot()
    def on_ws_disconnected(self):
        self.close_websocket()
        
    @Slot()
    def on_ws_error(self, error):
        err_msg = self.ws_client.errorString() or "WebSocket connection error"
        self.update_metrics_display({"WS Error": err_msg})
        self.close_websocket()

    def close_websocket(self):
        if self.ws_client.isValid():
            self.ws_client.close()

        self.is_tracking = False
        self.stop_video_worker()

        if self.encoder_worker:
            self.encoder_worker.stop()
            self.encoder_worker = None

        self.btn_track.setText("Track")

    def send_encoded_frame(self, data: bytes):
        if not self.ws_client.isValid():
            return
        if self.ws_client.bytesToWrite() > self.ws_max_buffer:
            return  # Drop frame — back-pressure
        self.ws_client.sendBinaryMessage(QByteArray(data))

    # ------------------------------------------------------------------
    # Video playback finished
    # ------------------------------------------------------------------

    @Slot()
    def on_video_finished(self):
        # Guard: ignore the spurious finished() fired by stop_video_worker()
        # during intentional teardown (e.g. Clear, source switch, close). 
        if self._stopping_video:
            return
        self.close_websocket()
        self.is_tracking = False
        self.is_frozen   = True
        self.btn_track.setText("Track")
        self.update_metrics_display({"Status": "Video Finished"})

        if self.encoder_worker:
            self.encoder_worker.stop()
            self.encoder_worker = None

        self.update_metrics_display({"Status": "Video Finished"})

    # ------------------------------------------------------------------
    # Follow
    # ------------------------------------------------------------------

    def trigger_follow(self):
        request = QNetworkRequest(QUrl(API_FOLLOW))
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )
        reply = self.network_manager.post(
            request, json.dumps({"action": "follow"}).encode()
        )
        reply.setProperty("req_type", "follow")

    @Slot(QNetworkReply)
    def on_http_response(self, reply: QNetworkReply):
        status_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if reply.error() == QNetworkReply.NetworkError.NoError and status_code == 200:
            try:
                data = json.loads(reply.readAll().data().decode())
                if "bbox" in data:
                    self.video_canvas.set_bbox(data["bbox"])
                self.update_metrics_display(data)
            except json.JSONDecodeError:
                self.update_metrics_display({"Error": "Invalid JSON response from server"})
        else:
            self.update_metrics_display({"Error": f"HTTP {status_code or 'Network Failure'}"})
        reply.deleteLater()

    # ------------------------------------------------------------------
    # Misc slots
    # ------------------------------------------------------------------

    def on_canvas_point_clicked(self, x: int, y: int):
        pass  # Reserved for future use

    def select_reference_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.lbl_ref_path.setText(path)
            with open(path, 'rb') as f:
                self.ref_image_bytes = f.read()

    def clear_all(self):
        self.close_websocket()
        self.is_frozen = False
        self.video_canvas.clear_overlays()
        self.update_metrics_display({"Status": "Idle"})

    def closeEvent(self, event):
        self.close_websocket()
        self.stop_video_worker()
        self.camera_thread.stop()
        event.accept()

# ----------------------------------------------------------------------

if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())