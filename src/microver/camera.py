import json
from collections.abc import Generator
from pathlib import Path

import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from waitress import serve

from microver.model import Model


class CameraRelay:
    def __init__(self, resolution=(400, 300)) -> None:
        self.picam = Picamera2()
        self.config = self.picam.create_video_configuration(main={"size": resolution})
        self.picam.configure(self.config)
        self.picam.start()

        self.app = Flask(__name__)
        self._setup_routes()

        self.models_path = Path(__file__).parent.parent / "models"
        with open(self.models_path / "coco_classes.json", "r") as file:
            self.classes = [None] + list(json.load(file).values())

        model_path = self.models_path / "ssd_mobilenet_v1_12.onnx"
        self.model = Model(str(model_path), self.classes)

    def get_frames(self) -> Generator[bytes, None, None]:
        while True:
            frame = self.picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = self.model.predict(frame)
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield buffer.tobytes()

    def _setup_routes(self) -> None:
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

    def close(self) -> None:
        self.picam.close()

    def serve(self) -> None:
        try:
            print("Starting camera relay server on http://10.124.0.129:3000")
            serve(self.app, host="0.0.0.0", port=3000, threads=2)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting program.")
        finally:
            self.close()
