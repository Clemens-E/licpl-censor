import supervision as sv
import numpy as np


def resize_boxes(boxes: np.ndarray, scale_factor: float) -> np.ndarray:
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    new_sizes = (boxes[:, 2:] - boxes[:, :2]) * scale_factor
    new_boxes = np.concatenate((centers - new_sizes / 2, centers + new_sizes / 2), axis=1)
    return new_boxes

def getDetectionsFromFrames(yol, frames):

    rawDe = yol.predict(source=frames, imgsz=1920, verbose=False, conf=0.15)
    detectionsList = []
    for frame in rawDe:
        detections = sv.Detections.from_ultralytics(frame)
        detections.xyxy = resize_boxes(detections.xyxy, 1.5)
        detectionsList.append(detections) 
    del rawDe
    return detectionsList
