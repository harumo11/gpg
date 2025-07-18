import numpy as np
import open3d as o3d
from darboux_frame import compute_darboux_frame_for_points
from darboux_frame import generate_hand_pose_candidates
from gripper import Gripper

class PoseCandidateGenerator:
    def __init__(self):
        self.gripper = Gripper(closure_volume_size=[0.05, 0.03, 0.05], finger_thickness=0.01)

    def _sample_random_points(self, pcd: o3d.geometry.PointCloud, N: int):
        """
        Sample N random points from a point cloud.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            The input point cloud from which to sample points.
        N : int
            The number of random points to sample.
        """

        # Check if the point cloud has enough points
        if len(pcd.points) < N:
            raise ValueError("The point cloud has fewer points than requested for sampling.")

        # Randomly sample N indices from the point cloud
        indices = np.random.choice(len(pcd.points), size=N, replace=False)

        # Select the sampled points using the indices
        sampled_points = pcd.select_by_index(indices)

        return sampled_points

    def _get_hand_pose_candidates(self, pcd: o3d.geometry.PointCloud, sample_points: o3d.geometry.PointCloud, y_offsets: np.ndarray | None, phi_angles: np.ndarray | None):
        """
        Generate hand pose candidates based on the base frame and offsets.
        """
        darboux_frames = compute_darboux_frame_for_points(pcd, sample_points)

        if y_offsets is None:
            y_offsets = np.linspace(-0.02, 0.02, 8)
        if phi_angles is None:
            phi_angles = np.linspace(-np.pi/2, np.pi/2, 8)

        hand_pose_candidates = []
        for frame in darboux_frames:
            if frame is None:
                continue
            poses = generate_hand_pose_candidates(frame, y_offsets, phi_angles)
            hand_pose_candidates.extend(poses)

        return hand_pose_candidates
    
    def generate_candidates(self, pcd: o3d.geometry.PointCloud, N: int = 1000, y_offsets: np.ndarray | None = None, phi_angles: np.ndarray | None = None):
        """
        Generate hand pose candidates from a point cloud.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            The input point cloud.
        N : int, optional
            The number of random points to sample from the point cloud, by default 1000.
        y_offsets : np.ndarray | None, optional
            Offsets in the y-direction for hand pose generation, by default None.
        phi_angles : np.ndarray | None, optional
            Angles for hand pose generation, by default None.

        Returns
        -------
        list
            A list of generated hand pose candidates.
        """
        sampled_points = self._sample_random_points(pcd, N)
        return self._get_hand_pose_candidates(pcd, sampled_points, y_offsets, phi_angles)