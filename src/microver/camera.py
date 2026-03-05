import importlib.resources as pkg_resources

import cv2
import numpy as np
import onnxruntime as rt
from flask import Flask, Response
from picamera2 import Picamera2
from waitress import serve

import microver.models


class CameraRelay:
    def __init__(self, resolution=(800, 600)) -> None:
        self.picam = Picamera2()
        self.config = self.picam.create_video_configuration(main={"size": resolution})
        self.picam.configure(self.config)
        self.picam.start()

        self.app = Flask(__name__)
        self._setup_routes()

        with pkg_resources.path(
            microver.models, "ssd_mobilenet_v1_13-qdq.onnx"
        ) as model_path:
            self.session = rt.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )

        self.input_name = self.session.get_inputs()[0].name
        self.classes = [
            "background",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
        ]

    def detect(self, frame):
        height, width, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = np.expand_dims(rgb.astype(np.uint8), axis=0)

        outputs = [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ]

        num, boxes, scores, classes = self.session.run(outputs, {self.input_name: data})
        num = int(num[0])  # pyright: ignore

        for i in range(num):
            score = scores[0][i]  # pyright: ignore

            if score < 0.5:
                continue

            box = boxes[0][i]  # pyright: ignore

            cls = int(classes[0][i])  # pyright: ignore

            top = int(box[0] * height)
            left = int(box[1] * width)
            bottom = int(box[2] * height)
            right = int(box[3] * width)

            label = self.classes[cls] if cls < len(self.classes) else str(cls)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame

    def get_frames(self):
        while True:
            frame = self.picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = self.detect(frame)
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield buffer.tobytes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return "<img src='/data'>"

        @self.app.route("/data")
        def get_data():
            return Response(
                (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    for frame in self.get_frames()
                ),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def close(self):
        self.picam.close()

    def serve(self):
        try:
            print("Starting camera relay server on http://10.124.0.129:3000")
            serve(self.app, host="0.0.0.0", port=3000, threads=2)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting program.")
        finally:
            self.close()
