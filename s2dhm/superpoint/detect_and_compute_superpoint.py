import cv2
import sys
import numpy as np

def superpoint_preprocess(image_path):
    """Apply SuperPoint Pre-processing."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype('float')/255.0
    return image.astype('float32')

def detect_and_compute(image_path, superpoint_network, to_cv2=False):
    """Detect and compute SuperPoint features."""
    input_image = superpoint_preprocess(image_path)
    keypoints, desc, scores = superpoint_network.run(input_image)
    if(to_cv2):
        keypoints = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts.T]
    return keypoints, desc, scores
