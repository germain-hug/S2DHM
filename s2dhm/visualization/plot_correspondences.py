import gin
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


@gin.configurable
def plot_image_retrieval(left_image_path: str,
                         right_image_path: str,
                         title: str,
                         export_folder: str,
                         export_filename: str):
    """Display (query, nearest-neighbor) pairs of images."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]
    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

@gin.configurable
def plot_correspondences(left_image_path: str,
                         right_image_path: str,
                         left_keypoints: List[cv2.KeyPoint],
                         right_keypoints: List[cv2.KeyPoint],
                         matches: np.ndarray,
                         title: str,
                         export_folder: str,
                         export_filename: str):
    """Display feature correspondences."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]

    # Draw Lines and Points
    for m in matches:
        left = left_keypoints[m[0]].pt
        right = tuple(sum(x) for x in zip(
            right_keypoints[m[1]].pt, (left_image.shape[1], 0)))
        cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255), 2)

    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

@gin.configurable
def plot_detections(left_image_path: str,
                    right_image_path: str,
                    left_keypoints: np.ndarray,
                    right_keypoints: np.ndarray,
                    title: str,
                    export_folder: str,
                    export_filename: str):
    """Display Superpoint detections."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)

    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    offset = left_image.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]

    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.scatter(
        left_keypoints.T[:, 0], left_keypoints.T[:, 1], c='red', s=5)
    plt.scatter(
        right_keypoints.T[:, 0] + offset, right_keypoints.T[:, 1], c='red', s=5)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)
