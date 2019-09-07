import gin
import numpy as np
from glob import glob
from pathlib import Path
from datasets.base_dataset import BaseDataset
from typing import List

@gin.configurable
class ExtendedCMUDataset(BaseDataset):
    """Extended CMU Dataset Loader.
    """
    def __init__(self, root: str,
                       image_folder: str,
                       reference_sequences: List[str],
                       query_sequences: List[str],
                       nvm_model: str=None,
                       binary_model: str=None,
                       bundler_out_model: str=None,
                       bundler_txt_model: str=None,
                       triangulation_data_file: str=None,
                       name: str=None,
                       database_folder: str=None,
                       queries_folder: str=None,
                       cmu_slice: int=None):
        """Initialize CMU class attributes.

        Args:
            root: The root to the dataset folder.
            image_folder: The folder containing the images.
            reference_sequences: The name of the reference sequences (should
                match the folder names).
            query_sequences: The name of the query sequences (should
                match the folder names).
            nvm_model: The path to a NVM reconstruction from VisualSfM.
            binary_model: The path to a binary COLMAP reconstruction file.
            bundler_out_model: The path to a bundler .out file.
            bundler_txt_model: The path to the matching bundler .list.txt file.
            triangulation_data: The path to the triangulated .npz file.
            name: The dataset name.
            database_folder: The subfolder name containing the database images.
            queries_folder: The subfolder name containing the query images.
            cmu_slice: The index of the CMU slice.
        """
        binary_model = str(Path(root, 'slice{}'.format(cmu_slice), 'sparse/'))
        triangulation_data_file = \
            '../data/triangulation/cmu_triangulation_slice_{}.npz'.format(cmu_slice)
        self._data = {
            'name': 'cmu',
            'root': root,
            'image_folder': image_folder,
            'reference_sequences': reference_sequences,
            'query_sequences': query_sequences,
            'nvm_model': nvm_model,
            'binary_model': binary_model,
            'bundler_out_model': bundler_out_model,
            'bundler_txt_model': bundler_txt_model,
            'triangulation_data_file': triangulation_data_file,
            'database_folder': database_folder,
            'queries_folder': queries_folder
        }
        self._cmu_slice = cmu_slice
        self.load_reference_image_names()
        self.load_query_image_names()
        self._pick_cmu_slice(cmu_slice)

        self.load_reference_camera_poses()
        self.load_triangulation_data()
        self.load_intrinsics()

    def _slice_to_index(self, cmu_slice):
        """Convert CMU slice to index in folders list."""
        cmu_slices = self._data['reference_sequences']
        cmu_slice_index = cmu_slices.index('slice{}'.format(cmu_slice))
        return cmu_slice_index

    def _pick_cmu_slice(self, cmu_slice):
        """Pick a single CMU slice."""
        cmu_slice_index = self._slice_to_index(cmu_slice)
        self._data['query_image_names'] = \
            self._data['query_image_names'][cmu_slice_index]
        self._data['reference_image_names'] = \
            self._data['reference_image_names'][cmu_slice_index]
        print('>> Found {} query images and {} reference images in slice {}.'.format(
            len(self._data['query_image_names']),
            len(self._data['reference_image_names']),
            cmu_slice))

    def load_reference_image_names(self):
        """Return all reference images.
        """
        image_roots = [Path(self._data['root'], self._data['image_folder'], f)
            for f in self._data['reference_sequences']]
        reference_images = [glob(str(Path(r, self._data['database_folder'], '*.jpg')))
            for r in image_roots]
        if not len(reference_images):
            path = str(Path(self._data['root'], self._data['image_folder']))
            raise Exception('No reference image found at {}'.format(path))
        print('>> Found {} reference slices.'.format(len(reference_images)))
        self._data['reference_image_names'] = reference_images

    def load_query_image_names(self):
        """Return all query images.
        """
        image_roots = [Path(self._data['root'], self._data['image_folder'], f)
            for f in self._data['query_sequences']]
        query_images = [glob(str(Path(r, self._data['queries_folder'], '*.jpg')))
            for r in image_roots]
        if not len(query_images):
            path = str(Path(self._data['root'], self._data['image_folder']))
            raise Exception('No query image found at {}'.format(path))
        print('>> Found {} query slices.'.format(len(query_images)))
        self._data['query_image_names'] = query_images

    def load_intrinsics(self):
        """Load query images intrinsics.

        For CMU, all images are rectified but have different intrinsics
            depending on the camera used.
        Returns:
            filename_to_intrinsics: A dictionary mapping a query filename to
                the intrinsics matrix and distortion coefficients.
        """
        f0 = (868.99 + 866.06) / 2.0
        f1 = (873.38 + 876.49) / 2.0
        K0 = np.array([[f0, 0.0, 525.942323],
                [0.0, f0, 420.042529],
                [0.0, 0.0, 1.0]], dtype=np.float32)
        K1 = np.array([[f1, 0.0, 529.324138],
                [0.0, f1, 397.272397],
                [0.0, 0.0, 1.0]], dtype=np.float32)
        d0 = np.array([-0.399431, 0.188924, 0.000153, 0.000571])
        d1 = np.array([-0.397066, 0.181925, 0.000176, -0.000579])
        filename_to_intrinsics = dict()
        for q in self._data['query_image_names']:
            if 'c0' in q:
                filename_to_intrinsics[q] = (K0, d0)
            elif 'c1' in q:
                filename_to_intrinsics[q] = (K1, d1)
        self._data['filename_to_intrinsics'] = filename_to_intrinsics

    def key_converter(self, filename: str):
        """Convert an absolute filename the keys format in the 3D files."""
        return '/'.join(filename.split('/')[-1:])

    def output_converter(self, filename: str):
        """Convert an absolute filename the output prediction format."""
        return '/'.join(filename.split('/')[-1:])

    @property
    def cmu_slice(self):
        return self._cmu_slice
