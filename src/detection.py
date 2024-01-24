import threading
from ultralytics import YOLO
import supervision as sv
import numpy as np


def getDetectionsBasedOnArgs(yol, frames, args):
    if args.multi:
        return detectFramesMulti(frames=frames, threadsToUse=args.threads)
    else:
        return detectFramesSingle(yol=yol, frames=frames)


def detectFramesSingle(yol, frames):
    rawDe = yol.predict(source=frames, imgsz=1920, verbose=False, conf=0.15)
    detectionsList = []
    for frame in rawDe:
        detections = sv.Detections.from_ultralytics(frame)
        detections.xyxy = resize_boxes(detections.xyxy, 1.5)
        detectionsList.append(detections)
    del rawDe
    return detectionsList


def detectFramesMulti(frames, threadsToUse: int):
    returnedDetections = [None] * len(frames)
    threads = []
    sem = threading.Semaphore()

    # add an index to each frame
    for i in range(len(frames)):
        frames[i] = (i, frames[i])
    # split frames into evenly chunks
    chunkedFrames = [frames[i::threadsToUse] for i in range(threadsToUse)]

    for i in range(threadsToUse):
        t = threading.Thread(target=thread_detectFrames,
                             args=(chunkedFrames[i], sem, returnedDetections))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()
    return returnedDetections


def resize_boxes(boxes: np.ndarray, scale_factor: float) -> np.ndarray:
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    new_sizes = (boxes[:, 2:] - boxes[:, :2]) * scale_factor
    new_boxes = np.concatenate(
        (centers - new_sizes / 2, centers + new_sizes / 2), axis=1)
    return new_boxes


def thread_detectFrames(frames, sem, results, args):
    model = YOLO(model=args.model)
    processedFrames = []
    for frame in frames:
        predicted = model.predict(source=frame[1],
                                  imgsz=1920,
                                  verbose=False,
                                  conf=0.15)[0]
        detc = sv.Detections.from_ultralytics(predicted)
        detc.xyxy = resize_boxes(detc.xyxy, 1.5)
        processedFrames.append((frame[0], detc))

    sem.acquire(timeout=10)

    for frame in processedFrames:
        results[frame[0]] = frame[1]

    sem.release()
