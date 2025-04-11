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
    result = []

    for i in range(len(detections)):
        # Get the frame range to consider (respecting list boundaries)
        start_idx = max(0, i - memorySize)
        end_idx = min(len(detections), i + memorySize + 1)

        # If current frame has no detections, just use an empty detection
        if i >= len(detections) or not detections[i] or len(
                detections[i]) == 0:
            result.append(sv.Detections.empty())
            continue

        # Collect all xyxy boxes, confidences, class_ids, and tracker_ids from the frame range
        all_xyxy = []
        all_confidence = []
        all_class_id = []
        all_tracker_id = []

        # Process frames in the memory window
        for j in range(start_idx, end_idx):
            if j >= len(detections) or detections[j] is None or len(
                    detections[j]) == 0:
                continue

            # Add all detections from this frame
            all_xyxy.extend(detections[j].xyxy.tolist())

            # Handle optional attributes
            if detections[j].confidence is not None:
                all_confidence.extend(detections[j].confidence.tolist())
            else:
                all_confidence.extend([1.0] * len(detections[j].xyxy))

            if detections[j].class_id is not None:
                all_class_id.extend(detections[j].class_id.tolist())
            else:
                all_class_id.extend([0] * len(detections[j].xyxy))

            if hasattr(detections[j],
                       'tracker_id') and detections[j].tracker_id is not None:
                all_tracker_id.extend(detections[j].tracker_id.tolist())

        # If no detections were found in the memory window
        if not all_xyxy:
            result.append(sv.Detections.empty())
            continue

        # Convert to numpy arrays
        np_xyxy = np.array(all_xyxy)
        np_confidence = np.array(all_confidence)
        np_class_id = np.array(all_class_id)

        # Use directly the sv.Detections.merge function or implement custom NMS
        # Option 1: Implement a custom NMS
        # Get indices of boxes to keep after NMS
        kept_indices = []

        # Convert boxes to [x1, y1, x2, y2] format for IoU calculation
        boxes = np_xyxy  # Already in the right format

        # Sort by confidence
        order = np.argsort(-np_confidence)

        while order.size > 0:
            # Pick the box with highest confidence
            i = order[0]
            kept_indices.append(i)

            # Calculate IoU of the picked box with rest
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # Calculate area of boxes
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                boxes[order[1:], 3] - boxes[order[1:], 1])

            # Calculate IoU
            union = area_i + area_others - intersection
            iou = intersection / union

            # Remove boxes with IoU > threshold
            inds = np.where(iou <= 0.3)[0]
            order = order[inds + 1]

        # Create new Detections object with NMS applied
        merged_xyxy = np_xyxy[kept_indices]
        merged_confidence = np_confidence[kept_indices]
        merged_class_id = np_class_id[kept_indices]

        # Create tracker_id array if it exists
        merged_tracker_id = None
        if all_tracker_id:
            np_tracker_id = np.array(all_tracker_id)
            merged_tracker_id = np_tracker_id[kept_indices]

        # Create the merged Detections object
        merged_detections = sv.Detections(xyxy=merged_xyxy,
                                          confidence=merged_confidence,
                                          class_id=merged_class_id,
                                          tracker_id=merged_tracker_id)

        result.append(merged_detections)

    return result


lastFrameIndex = 0


def getLowestConf(detc):
    proc = [d for d in detc]
    lowestConf = 1
    for detection in proc:
        print(detection[2])
        if detection[2] < lowestConf:
            lowestConf = detection[2]

    return lowestConf

def calculate_average_confidence(detections_list: list[sv.Detections]) -> float:
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
        if hasattr(detections, 'confidence') and detections.confidence is not None:
            # Add sum of confidences to total
            total_confidence += np.sum(detections.confidence)
            # Add number of detections to counter
            total_detections += len(detections.confidence)
    
    # Calculate and return average, or 0 if no detections
    if total_detections > 0:
        return total_confidence / total_detections
    else:
        return 0.0