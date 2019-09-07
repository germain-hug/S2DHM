"""Nearest Neighbor Pose Predictor Class.

This class predicts the query camera pose using the pose of the top-1 retrieved
image.
"""
import gin
from pose_prediction import predictor
from tqdm import tqdm


@gin.configurable
class NearestNeighborPredictor(predictor.PosePredictor):
    """Nearest Neighbor Pose Predictor Class.
    """
    def __init__(self, **kwargs):
        """Initialize base class attributes."""
        super().__init__(**kwargs)
        del self._network
        self._filename_to_pose = \
            self._dataset.data['reconstruction_data'].filename_to_pose

    def run(self):
        """Run the nearest neighbor pose predictor."""

        print('>> Generating pose predictions based on top-1 images...')
        output = []
        for i, rank in tqdm(enumerate(self._ranks.T), total=self._ranks.shape[1]):
            query_image = self._dataset.data['query_image_names'][i]
            for j in rank:
                nearest_neighbor = self._dataset.data['reference_image_names'][j]
                key = self._dataset.key_converter(nearest_neighbor)
                if key in self._filename_to_pose:
                    quaternion, camera_pose_matrix = self._filename_to_pose[key]
                    output.append(
                        [self._dataset.output_converter(query_image),
                        *quaternion, *list(camera_pose_matrix[:3,3])])
                    break
        return output
