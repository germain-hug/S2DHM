import os
import gin
import logging
import numpy as np
from pathlib import Path
from datasets import base_dataset
from network import network
from image_retrieval.pca import learn_and_apply_pca

def fetch_or_compute_ranks(dataset: base_dataset.BaseDataset,
                           network: network.ImageRetrievalModel):
    """Fetch pre-computed ranks for this dataset or compute them.
    """
    # Check if precomputed weights exist under
    dataset_name = dataset.data['name']
    if dataset.data['name']=='cmu':
        file_path = '../data/ranks/cmu/slice_{}.npz'.format(dataset.cmu_slice)
    else:
        file_path = '../data/ranks/{}.npz'.format(dataset.data['name'])

    if os.path.isfile(file_path):
        print('>> Found existing image ranks, loading {}'.format(file_path))
        return fetch_ranks(file_path, dataset, network)
    else:
        print('>> Computing image similarity ranks under {}'.format(file_path))
        return compute_ranks(file_path, dataset, network)

def fetch_ranks(file_path: str,
                dataset: base_dataset.BaseDataset,
                network: network.ImageRetrievalModel):
    """Retrieve pre-computed image ranks."""
    # Load numpy file
    numpy_file = np.load(file_path)

    # Check if reference and query images match the found images
    query_images = list(numpy_file['query_images'])
    reference_images = list(numpy_file['reference_images'])

    # If not, recompute them
    if (query_images != dataset.data['query_image_names'] or
        reference_images != dataset.data['reference_image_names']):
        logging.warn('Existing pre-computed file does not match dataset'
                     'configuration. Recomputing...')
        return compute_ranks(file_path, dataset, network)
    return numpy_file['ranks']

@gin.configurable
def compute_ranks(file_path: str,
                  dataset: base_dataset.BaseDataset,
                  network: network.ImageRetrievalModel,
                  use_pca: bool):
    """Compute image similarity."""
    # Compute reference images descriptors
    reference_images = dataset.data['reference_image_names']
    print('>> Computing descriptors for {} reference images'.format(
        len(reference_images)))
    reference_descriptors = network.compute_embedding(reference_images)

    # Compute query images descriptors
    query_images = dataset.data['query_image_names']
    print('>> Computing descriptors for {} query images'.format(
        len(query_images)))
    query_descriptors = network.compute_embedding(query_images)

    # (Optional) Learn and apply PCA
    if(use_pca):
        print('>> Learning and applying PCA...')
        reference_descriptors, query_descriptors = learn_and_apply_pca(
            reference_descriptors, query_descriptors)

    # Compute ranks
    print('>> Exporting results...')
    scores = np.dot(reference_descriptors, query_descriptors.T)
    ranks = np.argsort(-scores, axis=0)

    # Save filenames and ranks
    output_dict = {'ranks': ranks,
                   'query_images': query_images,
                   'reference_images': reference_images}
    Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    np.savez(file_path, **output_dict)
    return ranks
