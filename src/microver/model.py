from typing import TypeAlias, cast

import cv2
import numpy as np
import onnxruntime as rt

Box: TypeAlias = tuple[int, int, int, int]
Detection: TypeAlias = tuple[int, float, Box]
Detections: TypeAlias = list[Detection]


class Model:
    def __init__(self, model_path: str, classes: list[str] = []) -> None:
        print(f"Loading ONNX model from {model_path}...")

        self.options = rt.SessionOptions()
        self.options.intra_op_num_threads = 4
        self.options.inter_op_num_threads = 1

        self.session = rt.InferenceSession(
            model_path, sess_options=self.options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes = classes

        print("Model loaded successfully.")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0).astype(np.uint8)
        return image

    def inference(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outputs = [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ]

        result = self.session.run(outputs, {self.input_name: data})

        num = cast(np.ndarray, result[0])
        boxes = cast(np.ndarray, result[1])
        scores = cast(np.ndarray, result[2])
        classes = cast(np.ndarray, result[3])

        return num, boxes, scores, classes

    def postprocess(
        self,
        frame: np.ndarray,
        num: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        score_threshold=0.5,
    ) -> Detections:

        height, width = frame.shape[:2]
        detections: Detections = []

        n = int(num[0])

        for i in range(n):
            score = float(scores[0, i])

            if score < score_threshold:
                continue

            box = boxes[0, i]
            cls = int(classes[0, i])

            top = int(box[0] * height)
            left = int(box[1] * width)
            bottom = int(box[2] * height)
            right = int(box[3] * width)

            detections.append((cls, score, (left, top, right, bottom)))

        return detections

    def draw(
        self,
        frame: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:

        for cls, score, (left, top, right, bottom) in detections:
            label = (
                f"[{str(cls)}]: {self.classes[cls]}"
                if cls < len(self.classes)
                else str(cls)
            )

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (left, max(0, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame

    def predict(self, frame: np.ndarray) -> np.ndarray:
        data = self.preprocess(frame)
        num, boxes, scores, classes = self.inference(data)
        detections = self.postprocess(frame, num, boxes, scores, classes)
        return self.draw(frame, detections)
