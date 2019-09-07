import gin
import numpy as np
import datasets.internal
from glob import glob
from pathlib import Path
from datasets.base_dataset import BaseDataset


@gin.configurable
class RobotCarDataset(BaseDataset):
    """RobotCar Dataset Loader.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_intrinsics()

    def _glob_fn(self, root: str):
        """Glob pattern to fetch images."""
        return glob(str(Path(root, 'rear/*.jpg')))

    def load_intrinsics(self):
        """Load query images intrinsics.

        For RobotCar, all images are rectified and have the same intrinsics.
        Returns:
            filename_to_intrinsics: A dictionary mapping a query filename to
                the intrinsics matrix and distortion coefficients.
        """
        rear_intrinsics = np.reshape(np.array(
            [400.0, 0.0, 508.222931,
            0.0, 400.0, 498.187378,
            0.0, 0.0, 1], dtype=np.float32), (3, 3))
        distortion_coefficients = np.array([0.0,0.0,0.0,0.0])
        intrinsics = [(rear_intrinsics, distortion_coefficients)
            for i in range(len(self._data['query_image_names']))]
        filename_to_intrinsics = dict(
            zip(self._data['query_image_names'], intrinsics))
        self._data['filename_to_intrinsics'] = filename_to_intrinsics
