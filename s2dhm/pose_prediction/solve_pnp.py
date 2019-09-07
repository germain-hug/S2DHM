import gin
import cv2
import numpy as np
from collections import namedtuple
from pose_prediction import matrix_utils

Prediction = namedtuple('Prediction',
    'success num_matches num_inliers reference_inliers query_inliers quaternion matrix reference_filename reference_keypoints')


@gin.configurable
def solve_pnp(points_2D: np.ndarray,
              points_3D: np.ndarray,
              intrinsics: np.ndarray,
              distortion_coefficients: np.ndarray,
              reprojection_threshold: float,
              minimum_matches: int,
              minimum_inliers: int,
              reference_filename: str,
              reference_2D_points: np.ndarray,
              reference_keypoints: np.ndarray):
    """ Solve PnP using pre-established 2D-3D correspondences.

    Args:
        points_2D: The 2D points in the query image.
        points_3D: Their 3D match in the world frame.
        intrinsics: The intrinsics matrix.
        distortion_coefficients: The distortion_coefficients.
        reprojection_threshold: The RANSAC reprojection error threshold.
        minimum_matches: The minimum number of 2D-3D matches to run PnP.
        minimum_inliers: The minimum number of inliers for the PnP to be valid.
        reference_filename: The filename of the reference image (for
            visualization purposes).
        reference_2D_points: The reprojected 2D points in the reference image
            (for visualization purposes).
    Returns:
        Prediction: A prediction namedtuple.
    """
    # Run PnP + RANSAC if there are sufficient matches
    if points_2D.shape[0] > minimum_matches:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3D, points_2D, intrinsics, distortion_coefficients,
            iterationsCount=5000,
            reprojectionError=reprojection_threshold,
            flags=cv2.SOLVEPNP_P3P)
    else:
        return Prediction(
            success=False,
            num_matches=points_2D.shape[0],
            num_inliers=None,
            reference_inliers=None,
            query_inliers=None,
            quaternion=None,
            matrix=None,
            reference_filename=reference_filename,
            reference_keypoints=reference_keypoints)

    if success and len(inliers) >= minimum_inliers:
        success, rvec, tvec = cv2.solvePnP(
            points_3D[np.squeeze(inliers)],
            points_2D[np.squeeze(inliers)],
            intrinsics, distortion_coefficients, rvec=rvec, tvec=tvec,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        assert success
        matrix = matrix_utils.matrix_from_se3(tvec, rvec)
        quaternion = matrix_utils.matrix_quaternion(matrix)
        return Prediction(
            success=True,
            num_matches=points_2D.shape[0],
            num_inliers=len(inliers),
            reference_inliers=reference_2D_points[np.squeeze(inliers)],
            query_inliers=np.squeeze(points_2D[np.squeeze(inliers)]),
            quaternion=quaternion,
            matrix=matrix,
            reference_filename=reference_filename,
            reference_keypoints=reference_keypoints)
    else:
        return Prediction(
            success=False,
            num_matches=points_2D.shape[0],
            num_inliers=None,
            reference_inliers=None,
            query_inliers=None,
            quaternion=None,
            matrix=None,
            reference_filename=reference_filename,
            reference_keypoints=reference_keypoints)
