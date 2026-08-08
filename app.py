import sys
import json
import time
import cv2
from fractions import Fraction
import numpy as np
import av

from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl, QByteArray
from PySide6.QtGui import QImage, QPixmap
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
# Real-Time H.264 Encoder Helper using PyAV
# ----------------------------------------------------------------------
class H264Encoder:
    def __init__(self, width=640, height=480, fps=30, bitrate=1_000_000):
        self.codec = av.CodecContext.create('h264', 'w')
        self.codec.width = width
        self.codec.height = height
        self.codec.pix_fmt = 'yuv420p'
        self.codec.time_base = Fraction(1, fps)
        self.codec.bit_rate = bitrate
        self.codec.gop_size = 1  # Keyframe every frame for real-time tracking
        self.codec.options = {
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'repeat-headers': '1'
        }
        self.codec.open()

    def encode(self, cv_bgr_frame):
        if self.codec is None:
            return b""
        rgb_frame = cv2.cvtColor(cv_bgr_frame, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
        frame = frame.reformat(format='yuv420p')

        packets = self.codec.encode(frame)
        payload = bytearray()
        for packet in packets:
            payload.extend(bytes(packet))

        return bytes(payload)

    def close(self):
        """Flush remaining packets and close the encoder context safely."""
        if self.codec is not None:
            try:
                # Passing None to encode() flushes the encoder buffer
                packets = self.codec.encode(None)
                payload = bytearray()
                for packet in packets:
                    payload.extend(bytes(packet))
                self.codec = None
                return bytes(payload)
            except Exception:
                self.codec = None
        return b""


# ----------------------------------------------------------------------
# Interactive Video Canvas
# ----------------------------------------------------------------------
class VideoCanvas(QLabel):
    point_clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.click_enabled = True
        self.setStyleSheet("background-color: #121212; border: 1px solid #2a2a2a;")

        # Enable native Qt image scaling to avoid doing CPU-heavy scaling per frame
        self.setScaledContents(True)

        self.current_frame = None
        self.points = []
        self.last_bbox = None

    def set_frame(self, cv_img):
        self.current_frame = cv_img.copy()
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
        if event.button() == Qt.MouseButton.LeftButton and self.current_frame is not None:
            pixmap = self.pixmap()
            if not pixmap or pixmap.isNull():
                return

            p_w, p_h = pixmap.width(), pixmap.height()
            l_w, l_h = self.width(), self.height()
            x_offset, y_offset = (l_w - p_w) / 2, (l_h - p_h) / 2

            click_x = event.position().x() - x_offset
            click_y = event.position().y() - y_offset

            if 0 <= click_x <= p_w and 0 <= click_y <= p_h:
                frame_h, frame_w = self.current_frame.shape[:2]
                img_x = int((click_x / p_w) * frame_w)
                img_y = int((click_y / p_h) * frame_h)
                self.points.append((img_x, img_y))
                self.point_clicked.emit(img_x, img_y)
                self.update_display()

    def update_display(self):
        if self.current_frame is None:
            return

        display_img = self.current_frame.copy()
        orig_h, orig_w = display_img.shape[:2]

        # Draw Target Bounding Box
        if self.last_bbox and len(self.last_bbox) == 4:
            x, y, w, h = self.last_bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw Red Click Dots
        for pt in self.points:
            cv2.circle(display_img, pt, 4, (0, 0, 255), -1)

        # Direct pixmap assignment without per-frame software resampling
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_img.data, orig_w, orig_h, orig_w * 3, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img))


# ----------------------------------------------------------------------
# Camera Thread with Horizontal Mirroring
# ----------------------------------------------------------------------
class CameraWorker(QThread):
    frame_received = Signal(np.ndarray, float)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = True

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        failed_reads = 0

        while self.running and cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads > 50:
                    break
                self.msleep(10)
                continue
            failed_reads = 0
            frame = cv2.flip(frame, 1)
            latency = (time.time() - t0) * 1000
            self.frame_received.emit(frame, latency)
            self.msleep(1)

        cap.release()


# ----------------------------------------------------------------------
# Main Application Dashboard
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Dashboard")
        self.resize(1100, 650)

        # State Variables
        self.latest_raw_frame = None
        self.ref_image_bytes = None
        self.is_frozen = False
        self.is_tracking = False
        self.encoder = None

        # Metrics Dict Storage
        self.dynamic_metrics = {}

        # Qt Async HTTP & WebSockets
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_http_response)

        self.ws_client = QWebSocket()
        self.ws_client.connected.connect(self.on_ws_connected)
        self.ws_client.disconnected.connect(self.on_ws_disconnected)
        self.ws_client.textMessageReceived.connect(self.on_ws_message_received)

        self.init_ui()

        # Start Camera
        self.camera_thread = CameraWorker()
        self.camera_thread.frame_received.connect(self.on_frame_received)
        self.camera_thread.start()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_h_layout = QHBoxLayout(central_widget)

        # LEFT SIDE: Video Canvas & Controls
        left_v_layout = QVBoxLayout()

        self.video_canvas = VideoCanvas()
        self.video_canvas.point_clicked.connect(self.on_canvas_point_clicked)
        left_v_layout.addWidget(self.video_canvas, stretch=4)

        controls_group = QGroupBox("Control Panel")
        controls_v_layout = QVBoxLayout(controls_group)

        self.btn_capture = QPushButton("Capture Frame")
        self.btn_capture.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white; padding: 8px;")
        self.btn_capture.clicked.connect(self.capture_frame)
        controls_v_layout.addWidget(self.btn_capture)

        segment_box = QGroupBox("Segment Options")
        segment_v_layout = QVBoxLayout(segment_box)

        radio_layout = QHBoxLayout()
        self.radio_group = QButtonGroup(self)
        self.radio_click = QRadioButton("Click")
        self.radio_ref = QRadioButton("Reference")
        self.radio_text = QRadioButton("Text")
        self.radio_click.setChecked(True)

        self.radio_group.addButton(self.radio_click, 0)
        self.radio_group.addButton(self.radio_ref, 1)
        self.radio_group.addButton(self.radio_text, 2)

        radio_layout.addWidget(self.radio_click)
        radio_layout.addWidget(self.radio_ref)
        radio_layout.addWidget(self.radio_text)
        radio_layout.addStretch()

        segment_v_layout.addLayout(radio_layout)

        self.input_stack = QStackedWidget()
        self.input_stack.addWidget(QLabel("Click on the video stream above to select target points."))

        page_ref = QWidget()
        p_ref_l = QHBoxLayout(page_ref)
        p_ref_l.setContentsMargins(0, 0, 0, 0)
        self.btn_upload_ref = QPushButton("Upload Reference Image...")
        self.lbl_ref_path = QLabel("No file selected.")
        self.btn_upload_ref.clicked.connect(self.select_reference_image)
        p_ref_l.addWidget(self.btn_upload_ref)
        p_ref_l.addWidget(self.lbl_ref_path)
        p_ref_l.addStretch()
        self.input_stack.addWidget(page_ref)

        page_text = QWidget()
        p_txt_l = QHBoxLayout(page_text)
        p_txt_l.setContentsMargins(0, 0, 0, 0)
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText("Enter target text prompt...")
        p_txt_l.addWidget(self.txt_prompt)
        self.input_stack.addWidget(page_text)

        segment_v_layout.addWidget(self.input_stack)

        self.btn_run_segment = QPushButton("Segment Target")
        self.btn_run_segment.clicked.connect(self.trigger_segmentation)
        segment_v_layout.addWidget(self.btn_run_segment)

        controls_v_layout.addWidget(segment_box)

        act_layout = QHBoxLayout()
        self.btn_track = QPushButton("Track")
        self.btn_follow = QPushButton("Follow")
        self.btn_clear = QPushButton("Clear")

        self.btn_track.clicked.connect(self.toggle_tracking)
        self.btn_follow.clicked.connect(self.trigger_follow)
        self.btn_clear.clicked.connect(self.clear_all)

        act_layout.addWidget(self.btn_track)
        act_layout.addWidget(self.btn_follow)
        act_layout.addWidget(self.btn_clear)

        controls_v_layout.addLayout(act_layout)
        left_v_layout.addWidget(controls_group, stretch=1)

        main_h_layout.addLayout(left_v_layout, stretch=3)

        # RIGHT SIDE: Dynamic Metrics Panel
        right_metrics_box = QGroupBox("Backend Metrics")
        right_metrics_box.setMinimumWidth(300)

        self.metrics_container = QVBoxLayout(right_metrics_box)
        self.metrics_grid = QGridLayout()
        self.metrics_grid.setSpacing(10)

        self.metrics_container.addLayout(self.metrics_grid)
        self.metrics_container.addStretch()

        main_h_layout.addWidget(right_metrics_box, stretch=1)

        self.radio_group.idClicked.connect(self.on_mode_changed)
        self.update_metrics_display({"Status": "Idle"})

    def update_metrics_display(self, metrics_dict):
        while self.metrics_grid.count():
            item = self.metrics_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        row = 0
        for key, val in metrics_dict.items():
            k_lbl = QLabel(f"<b>{key}:</b>")
            v_lbl = QLabel(str(val))
            v_lbl.setWordWrap(True)

            self.metrics_grid.addWidget(k_lbl, row, 0)
            self.metrics_grid.addWidget(v_lbl, row, 1)
            row += 1

    @Slot(np.ndarray, float)
    def on_frame_received(self, frame, latency):
        self.latest_raw_frame = frame
        if not self.is_frozen:
            self.video_canvas.set_frame(frame)

        # Encode and send continuous raw H.264 NAL bytes
        if self.is_tracking and self.ws_client.isValid() and self.encoder:
            # Backpressure guard: drop frames if outgoing network buffer is accumulating
            if self.ws_client.bytesToWrite() < 200_000:
                h264_bytes = self.encoder.encode(frame)
                if h264_bytes:
                    self.ws_client.sendBinaryMessage(QByteArray(h264_bytes))

    def capture_frame(self):
        self.is_frozen = True
        self.video_canvas.points.clear()
        self.video_canvas.last_bbox = None
        self.video_canvas.update_display()
        self.update_metrics_display({"Status": "Frame Captured"})

    def trigger_segmentation(self):
        if self.latest_raw_frame is None:
            return

        method_id = self.radio_group.checkedId()
        method_map = {0: "click", 1: "reference", 2: "text"}
        method = method_map[method_id]

        multi_part = QHttpMultiPart(QHttpMultiPart.ContentType.FormDataType)

        method_part = QHttpPart()
        method_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, 'form-data; name="method"')
        method_part.setBody(method.encode('utf-8'))
        multi_part.append(method_part)

        # Single image payload remains JPEG over HTTP
        _, buffer = cv2.imencode('.jpg', self.latest_raw_frame)
        file_part = QHttpPart()
        file_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, 'form-data; name="file"; filename="frame.jpg"')
        file_part.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "image/jpeg")
        file_part.setBody(buffer.tobytes())
        multi_part.append(file_part)

        if method == "click":
            p_part = QHttpPart()
            p_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, 'form-data; name="points"')
            p_part.setBody(json.dumps(self.video_canvas.points).encode('utf-8'))
            multi_part.append(p_part)

        elif method == "reference" and self.ref_image_bytes:
            ref_part = QHttpPart()
            ref_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, 'form-data; name="ref_file"; filename="ref.jpg"')
            ref_part.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "image/jpeg")
            ref_part.setBody(self.ref_image_bytes)
            multi_part.append(ref_part)

        elif method == "text":
            t_part = QHttpPart()
            t_part.setHeader(QNetworkRequest.KnownHeaders.ContentDispositionHeader, 'form-data; name="text"')
            t_part.setBody(self.txt_prompt.text().strip().encode('utf-8'))
            multi_part.append(t_part)

        request = QNetworkRequest(QUrl(API_SEGMENT))
        reply = self.network_manager.post(request, multi_part)
        reply.setProperty("req_type", "segment")
        multi_part.setParent(reply)

    def on_mode_changed(self, idx):
        self.input_stack.setCurrentIndex(idx)
        self.video_canvas.click_enabled = (idx == 0)

        if idx != 0:
            self.video_canvas.points.clear()
            self.video_canvas.update_display()

    def toggle_tracking(self):
        if not self.is_tracking:
            self.ws_client.open(QUrl(WS_TRACK))
        else:
            self.close_websocket()

    @Slot()
    def on_ws_connected(self):
        bbox = self.video_canvas.last_bbox
        if bbox is None or self.latest_raw_frame is None:
            return

        self.video_canvas.points.clear()
        self.video_canvas.update_display()

        h, w = self.latest_raw_frame.shape[:2]
        self.encoder = H264Encoder(width=w, height=h, fps=30)

        # Step 1: BBox Metadata
        self.ws_client.sendTextMessage(json.dumps({"bbox": bbox}))

        # Step 2: First keyframe
        h264_bytes = self.encoder.encode(self.latest_raw_frame)
        if h264_bytes:
            self.ws_client.sendBinaryMessage(QByteArray(h264_bytes))

        # NOTE: Do NOT set self.is_tracking = True here yet! Wait for backend confirmation.

    @Slot(str)
    def on_ws_message_received(self, message):
        try:
            data = json.loads(message)
            if data.get("status") == "initialized":
                # Server is ready; begin frame streaming now
                self.is_tracking = True
                self.is_frozen = False
                self.btn_track.setText("Stop Track")
                return

            if "bbox" in data:
                self.video_canvas.set_bbox(data["bbox"])
            self.update_metrics_display(data)
        except json.JSONDecodeError:
            pass

    @Slot()
    def on_ws_disconnected(self):
        self.close_websocket()

    def close_websocket(self):
        if self.ws_client.isValid():
            self.ws_client.close()
        self.is_tracking = False

        # Flush and clear encoder context safely
        if self.encoder:
            self.encoder.close()
            self.encoder = None

        self.btn_track.setText("Track")

    def trigger_follow(self):
        request = QNetworkRequest(QUrl(API_FOLLOW))
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        payload = json.dumps({"action": "follow"}).encode('utf-8')
        reply = self.network_manager.post(request, payload)
        reply.setProperty("req_type", "follow")

    @Slot(QNetworkReply)
    def on_http_response(self, reply: QNetworkReply):
        if reply.error() != QNetworkReply.NetworkError.NoError:
            err_msg = reply.errorString()
            self.update_metrics_display({"Status": "Error", "Details": err_msg})
        else:
            try:
                data = json.loads(reply.readAll().data().decode('utf-8'))
                if "bbox" in data:
                    self.video_canvas.set_bbox(data["bbox"])
                self.update_metrics_display(data)
            except json.JSONDecodeError:
                pass
        reply.deleteLater()

    def on_canvas_point_clicked(self, x, y):
        pass

    def select_reference_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Ref Image", "", "Images (*.png *.jpg *.jpeg)")
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
        self.camera_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())