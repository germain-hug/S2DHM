"""Base Dataset Loader Class.
"""
import os
import sys
import pickle
import numpy as np
from glob import glob
from typing import List
from pathlib import Path
from collections import namedtuple
from datasets.model_parser import ModelParser

sys.path.insert(0, 'datasets/')


reconstruction_data = namedtuple('reconstruction_data',
    'intrinsics distortion_coefficients points_2D points_3D')


class BaseDataset():
    """Dataset Base Loader Class.
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
                       name: str=None):
        """Initialize base class attributes.

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
        """
        self._data = {
            'name': name,
            'root': root,
            'image_folder': image_folder,
            'reference_sequences': reference_sequences,
            'query_sequences': query_sequences,
            'nvm_model': nvm_model,
            'binary_model': binary_model,
            'bundler_out_model': bundler_out_model,
            'bundler_txt_model': bundler_txt_model,
            'triangulation_data_file': triangulation_data_file
        }
        self.load_reference_image_names()
        self.load_query_image_names()
        self.load_reference_camera_poses()
        self.load_triangulation_data()

    def _glob_fn(self, root: str):
        """Glob pattern to fetch images."""
        return glob(str(Path(root, '**/*.jpg')))

    def _assemble_intrinsics(self, focal, cx, cy, distortion):
        """Assemble intrinsics matrix from parameters."""
        intrinsics = np.eye(3)
        intrinsics[0,0] = float(focal)
        intrinsics[1,1] = float(focal)
        intrinsics[0,2] = float(cx)
        intrinsics[1,2] = float(cy)
        distortion_coefficients = np.array([float(distortion), 0.0, 0.0, 0.0])
        return intrinsics, distortion_coefficients

    def load_reference_image_names(self):
        """Load reference image names."""

        image_roots = [Path(self._data['root'], self._data['image_folder'], f)
            for f in self._data['reference_sequences']]
        reference_image_names = [self._glob_fn(r) for r in image_roots]
        reference_image_names = [i for s in reference_image_names for i in s]
        if not len(reference_image_names):
            path = str(Path(self._data['root'], self._data['image_folder']))
            raise Exception('No reference image found at {}'.format(path))
        print('>> Found {} reference images.'.format(
            len(reference_image_names)))
        self._data['reference_image_names'] = reference_image_names

    def load_query_image_names(self):
        """Load query image names."""

        image_roots = [Path(self._data['root'], self._data['image_folder'], f)
            for f in self._data['query_sequences']]
        query_image_names = [self._glob_fn(r) for r in image_roots]
        query_image_names = [i for s in query_image_names for i in s]
        if not len(query_image_names):
            path = str(Path(self._data['root'], self._data['image_folder']))
            raise Exception('No query image found at {}'.format(path))
        print('>> Found {} query images.'.format(len(query_image_names)))
        self._data['query_image_names'] = query_image_names

    def load_reference_camera_poses(self):
        """Load dataset ground truth camera poses from the reconstructions."""

        if self._data['nvm_model']:
            print('>> Loading poses from {}'.format(self._data['nvm_model']))
            self._data['reconstruction_data'] = ModelParser.from_nvm(
                self._data['nvm_model'])
        elif self._data['binary_model']:
            print('>> Loading poses from {}'.format(self._data['binary_model']))
            self._data['reconstruction_data'] = ModelParser.from_binary(
                self._data['binary_model'])
        elif self._data['bundler_out_model'] and self._data['bundler_txt_model']:
            print('>> Loading poses from {}'.format(
                self._data['bundler_out_model']))
            self._data['reconstruction_data'] = ModelParser.from_bundler(
                self._data['bundler_out_model'],
                self._data['bundler_txt_model'])
        else:
            pass
            raise Exception('No reconstruction model was provided! Please' \
                            'provide a link to a [.out|.nvm|.bin] file.')

    def load_triangulation_data(self):
        """ Load triangulation data.

        Returns:
            A dictionary mapping reference image filenames to triangulation
            data (intrinsics matrix, distortion coefficients, 2D and 3D points)
        """
        assert os.path.isfile(self._data['triangulation_data_file'])
        triangulation_data = np.load(self._data['triangulation_data_file'])
        filename_to_local_reconstruction = dict()
        for filename in self._data['reference_image_names']:
            key = self.key_converter(filename)
            if key in triangulation_data.files:
                local_reconstruction = triangulation_data[key].item()
                intrinsics, distortion_coefficients = self._assemble_intrinsics(
                    *local_reconstruction['K'].params)
                points_3D = local_reconstruction['points3D']
                points_2D = local_reconstruction['points2D']
                filename_to_local_reconstruction[filename] = \
                    reconstruction_data(intrinsics, distortion_coefficients,
                        points_2D, points_3D)
        self._data['filename_to_local_reconstruction'] = \
            filename_to_local_reconstruction

    def key_converter(self, filename: str):
        """Convert an absolute filename the keys format in the 3D files."""
        return '/'.join(filename.split('/')[-3:])

    def output_converter(self, filename: str):
        """Convert an absolute filename the output prediction format."""
        return '/'.join(filename.split('/')[-2:])

    @property
    def data(self):
        return self._data
