"""Base Pose Predictor Classes.
"""
import gin
import numpy as np
import pandas as pd
from datasets.base_dataset import BaseDataset
from network.network import ImageRetrievalModel
from pathlib import Path
from typing import List
from pose_prediction import solve_pnp
from visualization import plot_correspondences


@gin.configurable
class PosePredictor():
    """Pose Predictor Base Class.
    """
    def __init__(self, dataset: BaseDataset,
                       network: ImageRetrievalModel,
                       ranks: np.ndarray,
                       output_filename: str,
                       log_images: bool):
        """Initialize base class attributes.

        Args:
            dataset: The dataset loader containing ground truth data and query
                images.
            network: The image retrieval network, used to compute hypercolumns.
            ranks: The pre-computed image similarity ranking.
        """
        self._dataset = dataset
        self._ranks = ranks
        self._network = network
        self._output_filename = output_filename
        self._log_images = log_images

    def _choose_best_prediction(self, predictions, query_image):
        """Pick the best prediction from the top-N nearest neighbors."""
        filename = self._dataset.output_converter(query_image)
        best_prediction = np.argmax([p.num_inliers for p in predictions])
        quaternion = predictions[best_prediction].quaternion
        matrix = predictions[best_prediction].matrix
        return [filename, *quaternion, *list(matrix[:3,3])], predictions[best_prediction]

    def _nearest_neighbor_prediction(self, nearest_neighbor):
        key = self._dataset.key_converter(nearest_neighbor)
        if key in self._filename_to_pose:
            quaternion, matrix = self._filename_to_pose[key]
            prediction = solve_pnp.Prediction(
                success=True,
                num_matches=None,
                num_inliers=-1,
                reference_inliers=None,
                query_inliers=None,
                quaternion=quaternion,
                matrix=matrix,
                reference_filename=nearest_neighbor,
                reference_keypoints=None)
            return prediction
        return None

    def _plot_inliers(self, left_image_path, right_image_path, left_keypoints,
        right_keypoints, matches, title, export_filename):
        """Plot the inliers."""
        plot_correspondences.plot_correspondences(
            left_image_path=left_image_path,
            right_image_path=right_image_path,
            left_keypoints=left_keypoints,
            right_keypoints=right_keypoints,
            matches=matches,
            title=title,
            export_filename=export_filename)

    def save(self, predictions: List):
        """Export the predictions as a .txt file.

        Args:
            predictions: The list of predictions, where each line contains a
                [query name, quaternion, translation], as per the CVPR Visual
                Localization Challenge.
        """
        print('>> Saving predictions under {}'.format(self._output_filename))
        Path(self._output_filename).parent.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(np.array(predictions))
        df.to_csv(self._output_filename, sep=' ', header=None, index=None)
