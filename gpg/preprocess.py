import open3d as o3d

class Preprocess:
    def __init__(self):
        pass

    def __call__(self, pcd):
        """
        Process the point cloud to remove outliers.
        :param pcd: Open3D PointCloud object
        :return: Processed PointCloud object
        """
        # Remove outliers
        pcd, ind = self._remove_outliers(pcd)

        # Downsample the point cloud
        pcd = self._voxel_downsample(pcd, voxel_size=0.003)
        return pcd, ind
    
    def _remove_outliers(self, pcd):
        # Remove outliers using radius outlier removal
        #pcd, ind = pcd.remove_radius_outlier(nb_points=40, radius=0.05)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.5) # std_ratio gets smaller, more points are removed
        return cl, ind

    def _voxel_downsample(self, pcd, voxel_size=0.05):
        """
        Downsample the point cloud using voxel grid filtering.
        :param pcd: Open3D PointCloud object
        :param voxel_size: Size of the voxel grid
        :return: Downsampled PointCloud object
        """
        return pcd.voxel_down_sample(voxel_size=voxel_size)


def main():
    # Read the point cloud
    pcd = o3d.io.read_point_cloud("../data/desk_angledview.ply")
    
    # Create a processor instance
    processor = Preprocess()
    
    # Process the point cloud
    cl, indices = processor(pcd)
    inlier_pcd = pcd.select_by_index(indices)
    outlier_pcd = pcd.select_by_index(indices, invert=True)
    print(f'type of inlier_pcd: {type(inlier_pcd)}')

    # Paint the inliers and outliers with different colors
    inlier_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # gray
    outlier_pcd.paint_uniform_color([1, 0, 0])  # red
    
    # Display the processed point cloud
    #o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])
    o3d.visualization.draw_geometries([cl])

if __name__ == "__main__":
    main()
