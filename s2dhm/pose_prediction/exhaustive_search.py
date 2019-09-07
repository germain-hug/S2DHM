"""Helper functions to perform exhaustive search."""
import gin
import torch
import numpy as np
from typing import List


def retrieve_argmax(correspondence_map, factor,ratio=0.9):
    """Use a modified ratio test to obtain correspondences."""
    channels, width, height = correspondence_map.shape
    indices = torch.zeros((channels, 2))
    mask = torch.zeros((channels,))
    top_k = int(factor * width * height)
    for i, window in enumerate(correspondence_map):
        with torch.no_grad():
            dist_nn, ind = window.view(-1).topk(top_k, dim=-1, largest=True)
            match_ok = ((ratio ** 2) * dist_nn[0] >= dist_nn[-1])
            a_y, a_x = np.unravel_index(ind[0].item(), (width, height))
            indices[i,:] = torch.FloatTensor((a_x, a_y))
            mask[i] = match_ok.item()
    return indices, mask.cpu().numpy().astype(np.bool)

@gin.configurable
def exhaustive_search(dense_descriptor_map: torch.FloatTensor,
                      sparse_descriptor_map: torch.FloatTensor,
                      image_shape: List[int],
                      map_size: List[int],
                      cell_size: List[int],
                      factor: float):
    """ Perform exhaustive dense matching.

    Args:
        dense_descriptor_map: The dense query hypercolumn feature map.
        sparse_descriptor_map: The sparse reference set of hypercolumn.
        image_shape: The original image width and height.
        cell_size: The cell size in the reference image.
        factor: The thresholding factor for the ratio test.
    Returns:
        argmax_in_query_space: The set of corresponding points in query space.
        query_cell_sizes: The cell size in the query image.
        mask: The mask of valid correspondences after the ratio test.
    """
    # Compute correspondence maps map
    width, height = list(map(int, map_size))
    correspondence_map = torch.mm(
        sparse_descriptor_map, dense_descriptor_map).view(-1, width, height)
    # Find 2D maximums in correspondence map coordinates
    cell_sizes = torch.DoubleTensor(cell_size)
    indices, mask = retrieve_argmax(correspondence_map, factor)
    # Compute query argmax coordinates in image space
    map_size = torch.FloatTensor(list(correspondence_map.shape[1:]))
    image_shape = torch.FloatTensor(image_shape)
    query_cell_sizes = image_shape / map_size
    argmax_in_query_space = (indices * query_cell_sizes) + (query_cell_sizes / 2)
    return argmax_in_query_space, mask
