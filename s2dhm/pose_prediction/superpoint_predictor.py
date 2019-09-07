"""SuperPoint Detection and Matching Predictor Class.
"""
import gin
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pose_prediction import predictor
from pose_prediction import solve_pnp
from pose_prediction import keypoint_association
from pose_prediction import exhaustive_search
from superpoint import detect_and_compute_superpoint
from third_party.SuperPointPretrainedNetwork import demo_superpoint
from visualization import plot_correspondences


@gin.configurable
class SuperPointPredictor(predictor.PosePredictor):
    """SuperPoint Predictor Class.
    """
    def __init__(self, top_N: int, weights_path: str, nms_dist: int,
        conf_thresh: float, nn_thresh: float, ratio: float, **kwargs):
        """Initialize class attributes.

        Args:
            top_N: Number of nearest neighbors to consider in the
                sparse-to-dense matching.
        """
        super().__init__(**kwargs)
        self._top_N = top_N
        self._filename_to_pose = \
            self._dataset.data['reconstruction_data'].filename_to_pose
        self._filename_to_intrinsics = \
            self._dataset.data['filename_to_intrinsics']
        self._filename_to_local_reconstruction = \
            self._dataset.data['filename_to_local_reconstruction']
        self._superpoint_network = demo_superpoint.SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=torch.cuda.is_available())
        self._ratio = ratio

    def _assemble_matches(self, query_keypoints: np.ndarray,
                          keypoints: np.ndarray, points_2D: np.ndarray,
                          points_3D: np.ndarray, matches: np.ndarray):
        """ Build 2D-3D matches using the SuperPoint detections."""
        x2D, x2D_reference, x3D = [], [], []
        # Convert keypoints to cuda tensor for faster computation
        points_2D_cuda = torch.from_numpy(points_2D).to(self._network.device)
        torch_keypoints = torch.from_numpy(keypoints).to(self._network.device)
        argmins = keypoint_association.fast_closest_points(
            torch_keypoints[:2,:], points_2D_cuda)
        for i in range(torch_keypoints.shape[1]):
            if i in matches[:,1]:
                match_idx = list(matches[:,1]).index(i)
                match_in_I1 = matches[match_idx,0]
                x2D.append(query_keypoints.T[match_in_I1,:2])
                x3D.append(points_3D[argmins[i],:])
                x2D_reference.append(keypoints[:2,i].T)
        x2D = np.reshape(np.array(x2D), (-1, 1, 2))
        x3D = np.reshape(np.array(x3D), (-1, 1, 3))
        x2D_reference = np.reshape(np.array(x2D_reference), (-1, 1, 2))
        return x2D, x2D_reference, x3D

    def run(self):
        """Run the sparse-to-dense pose predictor."""

        print('>> Generating pose predictions using Superpoint detection and matching...')
        output = []
        tqdm_bar = tqdm(enumerate(self._ranks.T), total=self._ranks.shape[1],
                        unit='images', leave=True)
        for i, rank in tqdm_bar:

            # Compute the query superpoint descriptors
            query_image = self._dataset.data['query_image_names'][i]
            query_keypoints, query_descriptors, query_scores = \
                detect_and_compute_superpoint.detect_and_compute(
                    query_image, self._superpoint_network)
            predictions = []

            for j in rank[:self._top_N]:

                # Compute reference superpoint descriptors and perform matching
                nearest_neighbor = self._dataset.data['reference_image_names'][j]
                local_reconstruction = \
                    self._filename_to_local_reconstruction[nearest_neighbor]
                reference_keypoints, reference_descriptors, reference_scores = \
                    detect_and_compute_superpoint.detect_and_compute(
                        nearest_neighbor, self._superpoint_network)
                matches = keypoint_association.fast_keypoint_matching(
                    query_descriptors.T, reference_descriptors.T, self._ratio)

                # Assemble 2D-3D correspondences
                points_2D, points_2D_reference, points_3D = \
                self._assemble_matches(
                    query_keypoints,
                    reference_keypoints,
                    local_reconstruction.points_2D,
                    local_reconstruction.points_3D,
                    matches)

                # Solve PnP
                distortion_coefficients = \
                    local_reconstruction.distortion_coefficients
                intrinsics = local_reconstruction.intrinsics
                prediction = solve_pnp.solve_pnp(
                    points_2D=points_2D,
                    points_3D=points_3D,
                    intrinsics=intrinsics,
                    distortion_coefficients=distortion_coefficients,
                    reference_filename=nearest_neighbor,
                    reference_2D_points=points_2D_reference,
                    reference_keypoints=reference_keypoints)

                # If PnP failed, fall back to nearest-neighbor prediction
                if not prediction.success:
                    prediction = self._nearest_neighbor_prediction(
                        nearest_neighbor)
                    if prediction:
                        predictions.append(prediction)
                else:
                    predictions.append(prediction)

            if len(predictions):
                export, best_prediction = self._choose_best_prediction(
                    predictions, query_image)
                if self._log_images:
                    if np.ndim(np.squeeze(best_prediction.query_inliers)):
                        self._plot_inliers(
                            left_image_path=query_image,
                            right_image_path=best_prediction.reference_filename,
                            left_keypoints=keypoint_association.kpt_to_cv2(
                                np.squeeze(best_prediction.query_inliers)),
                            right_keypoints=keypoint_association.kpt_to_cv2(
                                np.squeeze(best_prediction.reference_inliers)),
                            matches=[(i, i) for i in range(best_prediction.num_inliers)],
                            title='SuperPoint Correspondences',
                            export_filename=self._dataset.output_converter(query_image))

                    if query_keypoints is not None and best_prediction.reference_keypoints is not None:
                        plot_correspondences.plot_detections(
                            left_image_path=query_image,
                            right_image_path=best_prediction.reference_filename,
                            left_keypoints=query_keypoints,
                            right_keypoints=best_prediction.reference_keypoints,
                            title='SuperPoint Detections',
                            export_filename=self._dataset.output_converter(query_image))

                    plot_correspondences.plot_image_retrieval(
                        left_image_path=query_image,
                        right_image_path=best_prediction.reference_filename,
                        title='SuperPoint Detections',
                        export_filename=self._dataset.output_converter(query_image))

                output.append(export)
                tqdm_bar.set_description(
                    "[{} inliers]".format(best_prediction.num_inliers))
                tqdm_bar.refresh()

        return output

    @property
    def dataset(self):
        return self._dataset
