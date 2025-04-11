import multiprocessing
import imageio
import supervision as sv
import numpy as np
import base64


def getFramesBufferMaxLength(args):
    return 30 if args.multi else args.batch


def getMaxCPUThreads():
    return multiprocessing.cpu_count()


def base64_encode_string(input_string):
    return base64.b64encode(input_string.encode()).decode()


def getFrameCount(inputFile):
    return imageio.get_reader(inputFile, "ffmpeg").count_frames()


def applyFrameMemory(detections: list[sv.Detections], memorySize: int):
    """
    Apply frame memory to detections by merging overlapping detections.
    
    Args:
        detections: List of sv.Detections objects for each frame
        memorySize: Number of frames to look ahead and behind for memory effect
        
    Returns:
        List of sv.Detections with merged overlapping detections for each frame
    """
    memoryFrames = []
    for index in range(len(detections)):
        mergedDetections = detections[index]

        for value2 in detections[index - memorySize:index + memorySize]:
            mergedDetections = sv.Detections.merge([mergedDetections, value2])

        if (index - memorySize) < 0:
            mergedDetections = sv.Detections.merge(
                [mergedDetections, detections[index]])

        mergedDetections = mergedDetections.with_nmm(threshold=0.3)
        memoryFrames.append(mergedDetections)
    return memoryFrames


lastFrameIndex = 0


def getLowestConf(detc):
    proc = [d for d in detc]
    lowestConf = 1
    for detection in proc:
        print(detection[2])
        if detection[2] < lowestConf:
            lowestConf = detection[2]

    return lowestConf


def calculate_average_confidence(
        detections_list: list[sv.Detections]) -> float:
    """
    Calculate the average confidence across a list of sv.Detections objects.
    
    Args:
        detections_list: List of sv.Detections objects
        
    Returns:
        float: Average confidence value across all detections, or 0 if no detections
    """
    # Initialize counters
    total_confidence = 0.0
    total_detections = 0

    # Iterate through each Detections object
    for detections in detections_list:
        # Skip if detections is None or empty
        if detections is None or len(detections) == 0:
            continue

        # Check if confidence attribute exists and is not None
        if hasattr(detections,
                   'confidence') and detections.confidence is not None:
            # Add sum of confidences to total
            total_confidence += np.sum(detections.confidence)
            # Add number of detections to counter
            total_detections += len(detections.confidence)

    # Calculate and return average, or 0 if no detections
    if total_detections > 0:
        return total_confidence / total_detections
    else:
        return 0.0
