import cv2
import torch
import numpy as np
from typing import List


def generate_dense_keypoints(feature_map_resolution: List[int],
                             image_resolution: List[int],
                             to_numpy : bool=True):
    """ Generate cell-based keypoints.

    Args:
        feature_map_resolution: The dense feature map [width, height].
        image_resolution: The reference image [width, height].
        to_numpy: Whether the outputted keypoints should be converted to numpy.
            Otherwise, they will be OpenCV Keypoints.
    Returns:
        keypoints: The dense set of keypoints computed on a regular grid.
        cell_size: The width and height of each cell in the grid.
    """
    # Extract cell sizes
    cell_size_x = image_resolution[0] / feature_map_resolution[0]
    cell_size_y = image_resolution[1] / feature_map_resolution[1]
    # Average (for cv2 Keypoints)
    cell_size = (cell_size_x + cell_size_y) / 2
    arange_x = np.arange(0, image_resolution[0], cell_size_x) + cell_size_x / 2
    arange_y = np.arange(0, image_resolution[1], cell_size_y) + cell_size_y / 2
    # Generate OpenCV KeyPoint objects
    keypoints = []
    for i in arange_x:
        for j in arange_y:
            if(to_numpy):
                keypoints.append([float(j), float(i)])
            else:
                keypoints.append(cv2.KeyPoint(float(j), float(i), cell_size))
    if to_numpy:
        keypoints = np.array(keypoints)
    return keypoints, (cell_size_x, cell_size_y)

def fast_sparse_keypoint_descriptor(keypoints, dense_keypoints,
                                    dense_descriptors):
    """ Associate keypoints with their nearest hypercolumn descriptor.

    Args:
        keypoints: List of keypoints of size [B x N x 2]
        dense_keypoints: Flattened dense keypoint grid of size [M x 2]
        dense_descriptors: Dense descriptor map of size [B x C x W x H]
    Returns:
        sparse_descriptors: B set of descriptors fetched at the keypoints
            locations.
    """
    # Flatten dense descriptors for faster matching
    batch_size, channels, width, height = dense_descriptors.shape
    dense_descriptors = dense_descriptors.view(batch_size, channels, -1)

    # Sparse hypercolumn descriptors associated with the sparse keypoints
    sparse_descriptors = [torch.zeros(
        (x.shape[1], channels)).cuda() for x in keypoints]

    # Associate each detected keypoint with the nearest dense descriptor
    for i, kp in enumerate(keypoints):
        # Find closest points between detected keypoints and dense keypoints
        argmins = fast_closest_points(
            torch.from_numpy(kp[:2,:]).cuda(), dense_keypoints)
        for j in range(kp.shape[1]):
            sparse_descriptors[i][j, :] = dense_descriptors[i, :, argmins[j]]
    return sparse_descriptors

def fast_closest_points(points, reference_points):
    """For each point in points, find the closest point in reference_points."""
    assert points.shape[0]==2
    assert reference_points.shape[1]==2
    k = points.shape[1]
    with torch.no_grad():
        reference_points = reference_points.unsqueeze(2).repeat(1,1,k)
        dist = torch.sum((points-reference_points)**2, dim=1)
        argmin = torch.argmin(dist, dim=0)
    return argmin

def fast_keypoint_matching(desc1, desc2, ratio_thresh, is_torch=False):
    '''A fast matching method that matches multiple descriptors simultaneously.
       Assumes that descriptors are normalized and can run on GPU if available.
       Performs the landmark-aware ratio test if labels are provided.
       From https://github.com/ethz-asl/hfnet
    '''
    if not is_torch:
        cuda = torch.cuda.is_available()
        desc1, desc2 = torch.from_numpy(desc1), torch.from_numpy(desc2)
        if cuda:
            desc1, desc2 = desc1.cuda(), desc2.cuda()
    with torch.no_grad():
        dist = 2*(1 - desc1 @ desc2.t())
        dist_nn, ind = dist.topk(2, dim=-1, largest=False)
        match_ok = (dist_nn[:, 0] <= (ratio_thresh**2)*dist_nn[:, 1])
        if match_ok.any():
            matches = torch.stack(
                [torch.nonzero(match_ok)[:, 0], ind[match_ok][:, 0]], dim=-1)
        else:
            matches = ind.new_empty((0, 2))
    return matches.cpu().numpy()

def kpt_to_cv2(kpts):
    """Convert an array of keypoints into cv2::KeyPoint objects."""
    return [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kpts]
