# Copied from https://github.com/roboflow/supervision/blob/develop/supervision/annotators/core.py
# significantly modified 
import numpy as np
import cv2
import math
from typing import Tuple

def clip_boxes(xyxy: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """
    Clips bounding boxes coordinates to fit within the frame resolution.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in
        the format `(x_min, y_min, x_max, y_max)`.
        resolution_wh (Tuple[int, int]): A tuple of the form `(width, height)`
            representing the resolution of the frame.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box with coordinates clipped to fit
            within the frame resolution.
    """
    result = np.copy(xyxy)
    width, height = resolution_wh
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result

class BlurAnnotator():
    """
    A class for blurring regions in an image using provided detections.
    """

    def __init__(self, blur_strength=30):
        """
        Initialize the blur annotator with customizable blur strength.
        
        Args:
            blur_strength (int): Controls the strength of the Gaussian blur.
                                Higher values create a stronger blur effect.
        """
        self.blur_strength = blur_strength

    def annotate(
        self,
        scene: np.ndarray,
        detections,
    ) -> np.ndarray:
        """
        Annotates the given scene by blurring regions based on the provided detections.

        Args:
            scene (np.ndarray): The image where blurring will be applied.
            detections (Detections): Object detections to annotate.

        Returns:
            The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...F
            >>> detections = sv.Detections(...)

            >>> blur_annotator = sv.BlurAnnotator()
            >>> annotated_frame = circle_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```
        """
        image_height, image_width = scene.shape[:2]
        clipped_xyxy = clip_boxes(
            xyxy=detections.xyxy, resolution_wh=(image_width, image_height)
        ).astype(int)

        for x1, y1, x2, y2 in clipped_xyxy:
            roi = scene[y1:y2, x1:x2]
            
            # Skip if ROI is empty
            if roi.size == 0:
                continue
                
            # Calculate kernel size based on region dimensions
            # but with a more moderate scaling for stability
            region_size = min(y2 - y1, x2 - x1)
            
            # Use a two-step blur for smoother results
            # First, a small box blur to reduce noise
            roi = cv2.boxFilter(roi, -1, (5, 5))
            
            # Then apply a Gaussian blur with size proportional to region
            # but capped to avoid excessive blurring
            blur_size = max(5, min(99, region_size // 3))
            # Ensure blur size is odd
            if blur_size % 2 == 0:
                blur_size += 1
                
            # Apply Gaussian blur with consistent sigma value for smoother effect
            sigma = self.blur_strength
            roi = cv2.GaussianBlur(roi, (blur_size, blur_size), sigma)
            
            # Apply the blurred region back to the image
            scene[y1:y2, x1:x2] = roi

        return scene