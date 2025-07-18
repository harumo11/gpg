import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_darboux_frame_for_points(pcd: o3d.geometry.PointCloud, sample_points: o3d.geometry.PointCloud, search_radius: float = 0.02):
    """
    Compute the Darboux frame for each point in pcd using sample_points.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The point cloud for which to compute the Darboux frame.
    sample_points : o3d.geometry.PointCloud
        The point cloud used for sampling to compute the Darboux frame.
    search_radius : float, optional
        The radius for neighborhood search used in PCA, by default 0.02[m].
    
    Returns
    -------
    darboux_frames : list
        A list of rotation matrices representing the Darboux frame for each point in pcd.
    """

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    sampled = np.asarray(sample_points.points)
    frames = []

    for p in sampled:
        [k, idx, _] = kdtree.search_radius_vector_3d(p, search_radius)
        if k < 3:
            frames.append(None) # Not enough neighbors to compute the frame
            print(f'||| warn: Point {p} has less than 3 neighbors, skipping.')
            continue

        neighbors = points[idx]
        # Compute the covariance matrix(centered)
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = centered.T @ centered

        # Compute the eigenvalues(ascending order) and eigenvectors(normalized) 
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Define the Darboux frame
        normal = eigvecs[:, 0] # Normal (Minimum eigenvalue)
        tangent1 = eigvecs[:, 2] # Principal curvature direction (Maximum eigenvalue)
        tangent2 = eigvecs[:, 1] # The other principal curvature direction (Middle eigenvalue)

        # Ensure normal looks to origin (assuming the sensor is at the origin (0, 0, 0))
        if np.dot(normal, p) < 0:
            normal = -normal

        # Ensure right-handed coordinate system
        if np.dot(np.cross(tangent1, tangent2), normal) < 0:
            tangent2 = -tangent2 # return tangent2 to become right-handedness

        # Create the rotation matrix
        rotation_matrix = np.column_stack((tangent1, tangent2, normal)) # x, y, z axes
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = p # Set the translation to the point p
        frames.append(T)

    return frames


def generate_hand_pose_candidates(base_frame: np.ndarray, y_offsets: np.ndarray | list, phi_angles: np.ndarray | list):
    """Generate grasp hand pose *h* by applying a grid of offsets/rotations to the base frame.

    Parameters
    ----------
    base_frame : np.ndarray
        Homogeneous transform that represents the Darboux frame located at the sample point.
        Columns correnspond to **x (approach)**, **y (closing)**, **z (normal)**.
    y_offsets : np.ndarray | list
        Offsets along the y-axis (closing direction) to apply to the base frame.
    phi_angles : np.ndarray | list
        Angles in radians to rotate the base frame around the z-axis (normal direction).
    
    Returns
    -------
    list[np.ndarray]
        List of homogeneous transforms representing the generated hand poses.

    """

    y_offsets = np.asarray(y_offsets).astype(float)
    phi_angles = np.asarray(phi_angles).astype(float)

    hand_poses = []
    for y_offset in y_offsets:
        for phi in phi_angles:
            transform = np.eye(4)
            transform[:3, :3] = R.from_euler('z', phi).as_matrix()  # Rotate around z-axis
            transform[:3, 3] = [0.0, y_offset, 0.0]
            hand_pose_candidate = base_frame @ transform  # Apply the transformation to the base frame
            hand_poses.append(hand_pose_candidate)
    
    return hand_poses