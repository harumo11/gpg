# Those functions are helper functions for implementation of High Precision Grasping in Dense Clutter

import open3d as o3d

def load_point_cloud(file_path: str='../data/table_topview.ply') -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a file.
    
    Args:
        file_path (str): Path to the point cloud file.
        
    Returns:
        o3d.geometry.PointCloud: Loaded point cloud.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"Failed to load point cloud from {file_path}")
    return pcd