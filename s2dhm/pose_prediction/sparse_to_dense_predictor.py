"""Sparse-To-Dense Predictor Class.
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
from visualization import plot_correspondences


@gin.configurable
class SparseToDensePredictor(predictor.PosePredictor):
    """Sparse-to-dense Predictor Class.
    """
    def __init__(self, top_N: int, **kwargs):
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

    def _compute_sparse_reference_hypercolumn(self, reference_image,
                                              local_reconstruction):
        """Compute hypercolumns at every visible 3D point reprojection."""
        reference_dense_hypercolumn, image_size = \
            self._network.compute_hypercolumn(
                [reference_image], to_cpu=False, resize=True)
        dense_keypoints, cell_size = keypoint_association.generate_dense_keypoints(
            (reference_dense_hypercolumn.shape[2:]),
            Image.open(reference_image).size[::-1], to_numpy=True)
        dense_keypoints = torch.from_numpy(dense_keypoints).cuda()
        reference_sparse_hypercolumns = \
            keypoint_association.fast_sparse_keypoint_descriptor(
                [local_reconstruction.points_2D.T],
                dense_keypoints, reference_dense_hypercolumn)[0]
        return reference_sparse_hypercolumns, cell_size

    def run(self):
        """Run the sparse-to-dense pose predictor."""

        print('>> Generating pose predictions using sparse-to-dense matching...')
        output = []
        tqdm_bar = tqdm(enumerate(self._ranks.T), total=self._ranks.shape[1],
                        unit='images', leave=True)
        for i, rank in tqdm_bar:

            # Compute the query dense hypercolumn
            query_image = self._dataset.data['query_image_names'][i]
            if query_image not in self._filename_to_intrinsics:
                continue
            query_dense_hypercolumn, _ = self._network.compute_hypercolumn(
                [query_image], to_cpu=False, resize=True)
            channels, width, height = query_dense_hypercolumn.shape[1:]
            query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view(
                (channels, -1))
            predictions = []

            for j in rank[:self._top_N]:

                # Compute dense reference hypercolumns
                nearest_neighbor = self._dataset.data['reference_image_names'][j]
                local_reconstruction = \
                    self._filename_to_local_reconstruction[nearest_neighbor]
                reference_sparse_hypercolumns, cell_size = \
                    self._compute_sparse_reference_hypercolumn(
                        nearest_neighbor, local_reconstruction)

                # Perform exhaustive search
                matches_2D, mask = exhaustive_search.exhaustive_search(
                    query_dense_hypercolumn,
                    reference_sparse_hypercolumns,
                    Image.open(nearest_neighbor).size[::-1],
                    [width, height],
                    cell_size)

                # Solve PnP
                points_2D = np.reshape(
                    matches_2D.cpu().numpy()[mask], (-1, 1, 2))
                points_3D = np.reshape(
                    local_reconstruction.points_3D[mask], (-1, 1, 3))
                distortion_coefficients = \
                    local_reconstruction.distortion_coefficients
                intrinsics = local_reconstruction.intrinsics
                prediction = solve_pnp.solve_pnp(
                    points_2D=points_2D,
                    points_3D=points_3D,
                    intrinsics=intrinsics,
                    distortion_coefficients=distortion_coefficients,
                    reference_filename=nearest_neighbor,
                    reference_2D_points=local_reconstruction.points_2D[mask],
                    reference_keypoints=None)

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
                                best_prediction.query_inliers),
                            right_keypoints=keypoint_association.kpt_to_cv2(
                                best_prediction.reference_inliers),
                            matches=[(i, i) for i in range(best_prediction.num_inliers)],
                            title='Sparse-to-Dense Correspondences',
                            export_filename=self._dataset.output_converter(query_image))

                    plot_correspondences.plot_image_retrieval(
                        left_image_path=query_image,
                        right_image_path=best_prediction.reference_filename,
                        title='Best match',
                        export_filename=self._dataset.output_converter(query_image))

                output.append(export)
                tqdm_bar.set_description(
                    "[{} inliers]".format(best_prediction.num_inliers))
                tqdm_bar.refresh()

        return output

    @property
    def dataset(self):
        return self._dataset
