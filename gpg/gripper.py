import open3d as o3d
import numpy as np
import copy


class Gripper:
    def __init__(self, closure_volume_size: list, finger_thickness: float = 0.01):
        """
        Initialize the Gripper object.

        Parameters
        ----------
        closure_volume_size : list
            Dimensions of the closure volume [width, height, depth].
        finger_thickness : float, optional
            Thickness of the gripper fingers, by default 0.01.
        """
        self._closure_volume_size = np.array(closure_volume_size)
        self._finger_size = [finger_thickness, self._closure_volume_size[1], self._closure_volume_size[2]]
        self._palm_size = [self._closure_volume_size[0], self._closure_volume_size[1], finger_thickness]
        self._transforms = self._create_transforms(self._finger_size, self._palm_size)
        self._meshes = self._create_meshes(self._finger_size, self._palm_size)
        self._bboxes = self._create_bboxes(self._closure_volume_size, self._transforms)

    def _create_bboxes(self, closure_volume_size, transforms):
        """
        Create bounding boxes for the gripper components.

        Parameters
        ----------
        closure_volume_size : array-like
            Dimensions of the closure volume [width, height, depth].
        transforms : dict
            Transformation matrices for the gripper components.

        Returns
        -------
        dict
            A dictionary containing the bounding box for the closure volume.
        """
        bboxes = {}
        min_bound = transforms['base_to_closure_volume_bottom'][:3, 3]
        max_bound = min_bound + closure_volume_size
        bboxes['closure_volume'] = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        return bboxes

    def _create_transforms(self, finger_size, palm_size):
        """
        Create transformation matrices for the gripper components.

        Parameters
        ----------
        finger_size : list
            Dimensions of the fingers [width, height, depth].
        palm_size : list
            Dimensions of the palm [width, height, depth].

        Returns
        -------
        dict
            A dictionary containing transformation matrices for the gripper components.
        """
        transforms = {}
        transforms['base_to_palm_bottom'] = np.eye(4)
        transforms['base_to_right_finger_bottom'] = np.eye(4)
        transforms['base_to_left_finger_bottom'] = np.eye(4)
        transforms['base_to_closure_volume_bottom'] = np.eye(4)

        # Palm transform
        transforms['base_to_palm_bottom'][0, 3] = -palm_size[0] / 2
        transforms['base_to_palm_bottom'][1, 3] = -palm_size[1] / 2
        transforms['base_to_palm_bottom'][2, 3] = -finger_size[2] - palm_size[2]
        transforms['base_to_palm_top'] = transforms['base_to_palm_bottom'].copy()
        transforms['base_to_palm_top'][2, 3] += palm_size[2]

        # Right finger transform
        transforms['base_to_right_finger_bottom'][0, 3] = palm_size[0] / 2
        transforms['base_to_right_finger_bottom'][1, 3] = -palm_size[1] / 2
        transforms['base_to_right_finger_bottom'][2, 3] = -finger_size[2]
        transforms['base_to_right_finger_top'] = transforms['base_to_right_finger_bottom'].copy()
        transforms['base_to_right_finger_top'][2, 3] += finger_size[2]

        # Left finger transform
        transforms['base_to_left_finger_bottom'][0, 3] = -palm_size[0] / 2 - finger_size[0]
        transforms['base_to_left_finger_bottom'][1, 3] = -palm_size[1] / 2
        transforms['base_to_left_finger_bottom'][2, 3] = -finger_size[2]
        transforms['base_to_left_finger_top'] = transforms['base_to_left_finger_bottom'].copy()
        transforms['base_to_left_finger_top'][2, 3] += finger_size[2]

        # Closure volume transform
        transforms['base_to_closure_volume_bottom'][0, 3] = -palm_size[0] / 2 + finger_size[0]
        transforms['base_to_closure_volume_bottom'][1, 3] = -palm_size[1]
        transforms['base_to_closure_volume_bottom'][2, 3] = -finger_size[2]
        return transforms

    def _has_collision(self, points: o3d.geometry.PointCloud) -> bool:
        """
        Check if the given points collide with the gripper's palm or fingers.

        Parameters
        ----------
        points : o3d.geometry.PointCloud
            The point cloud to check for collisions.

        Returns
        -------
        bool
            True if there is a collision, False otherwise.
        """
        collision_meshes = [self._meshes['palm'], self._meshes['right_finger'], self._meshes['left_finger']]
        for mesh in collision_meshes:
            bbox = mesh.get_axis_aligned_bounding_box()
            cropped_points = points.crop(bbox)
            if len(cropped_points.points) > 0:
                return True
        return False

    def _has_point_in_closure_volume(self, points: o3d.geometry.PointCloud) -> bool:
        """
        Check if the given points are inside the closure volume of the gripper.

        Parameters
        ----------
        points : o3d.geometry.PointCloud
            The point cloud to check.

        Returns
        -------
        bool
            True if points are inside the closure volume, False otherwise.
        """
        cropped_points = points.crop(self._bboxes['closure_volume'])
        return len(cropped_points.points) > 0

    def _get_maximum_movable_distance(self, points: o3d.geometry.PointCloud) -> float | None:
        """
        Calculate the maximum distance that the gripper can move along z-axis without collision.

        Parameters
        ----------
        points : o3d.geometry.PointCloud
            The point cloud to check for collisions.

        Returns
        -------
        float or None
            The maximum movable distance along the z-axis, or None if no collision is detected.
        """
        max_extend_distance = points.get_axis_aligned_bounding_box().get_extent()[2]

        def create_extended_bbox(transform, size):
            min_bound = transform[:3, 3]
            max_bound = min_bound + np.array([size[0], size[1], max_extend_distance])
            return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        extended_bboxes = {
            'right_finger': create_extended_bbox(self._transforms['base_to_right_finger_top'], self._finger_size),
            'left_finger': create_extended_bbox(self._transforms['base_to_left_finger_top'], self._finger_size),
            'palm': create_extended_bbox(self._transforms['base_to_palm_top'], self._palm_size)
        }

        nearest_z_point = None
        collided_part_name = None
        for name, bbox in extended_bboxes.items():
            cropped_points = points.crop(bbox)
            if len(cropped_points.points) > 0:
                z_coords = np.array(cropped_points.points)[:, 2]
                min_z = np.min(z_coords)
                if nearest_z_point is None or min_z < nearest_z_point:
                    nearest_z_point = min_z
                    collided_part_name = name

        if nearest_z_point is not None:
            if collided_part_name in ['right_finger', 'left_finger']:
                return nearest_z_point
            elif collided_part_name == 'palm':
                return nearest_z_point + self._finger_size[2]
        return None

    def crop_by_closure_volume(self, points: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Crop the given point cloud by the closure volume of the gripper.

        Parameters
        ----------
        points : o3d.geometry.PointCloud
            The point cloud to crop.

        Returns
        -------
        o3d.geometry.PointCloud
            The cropped point cloud.
        """
        return points.crop(self._bboxes['closure_volume'])

    def _create_meshes(self, finger_size, palm_size):
        """
        Create meshes for the gripper components.

        Parameters
        ----------
        finger_size : list
            Dimensions of the fingers [width, height, depth].
        palm_size : list
            Dimensions of the palm [width, height, depth].

        Returns
        -------
        dict
            A dictionary containing the meshes for the gripper components.
        """
        meshes = {}
        # Right finger mesh
        right_finger_mesh = o3d.geometry.TriangleMesh.create_box(*finger_size, create_uv_map=True)
        right_finger_mesh.translate(self._transforms['base_to_right_finger_bottom'][:3, 3])
        right_finger_mesh.paint_uniform_color([0.1, 0.4, 0.8])
        meshes['right_finger'] = right_finger_mesh

        # Left finger mesh
        left_finger_mesh = o3d.geometry.TriangleMesh.create_box(*finger_size)
        left_finger_mesh.translate(self._transforms['base_to_left_finger_bottom'][:3, 3])
        left_finger_mesh.paint_uniform_color([0.5, 0.1, 0.6])
        meshes['left_finger'] = left_finger_mesh

        # Palm mesh
        palm_mesh = o3d.geometry.TriangleMesh.create_box(*palm_size)
        palm_mesh.translate(self._transforms['base_to_palm_bottom'][:3, 3])
        palm_mesh.paint_uniform_color([0.4, 0.1, 0.3])
        meshes['palm'] = palm_mesh

        # Optionally, base frame mesh
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, center=[0, 0, 0])
        meshes['frame'] = frame_mesh
        return meshes
    
    def find_graspable_pose(self, candidate_pose: np.ndarray, points: o3d.geometry.PointCloud, show_debug: bool = False) -> np.ndarray | None:
        """
        Find a graspable candidate pose arround the given candidate pose.

        Parameters
        ----------
        candidate_pose : np.ndarray
            The candidate pose to check for graspability. This is homogeneous matrix [4, 4]
        points : o3d.geometry.PointCloud
            The point cloud to check for graspability.

        Returns
        -------
        gripper_pose : np.ndarray | None
            The graspable pose if found, otherwise None.
        """
        import copy

        if not isinstance(points, o3d.geometry.PointCloud):
            raise TypeError("points must be an instance of o3d.geometry.PointCloud")

        # Check if the candidate pose is valid
        if candidate_pose.shape != (4, 4):
            raise ValueError("candidate_pose must be a 4x4 homogeneous transformation matrix")

        # Transform the point cloud to the candidate pose go to the gripper frame (0, 0, 0)
        transformed_points = copy.deepcopy(points).transform(np.linalg.inv(candidate_pose))

        # Estimate the maximum movable distance along the z-axis
        max_distance = self._get_maximum_movable_distance(transformed_points)
        if max_distance is None:
            if show_debug:
                print(f"No collision detected, gripper can move freely. {max_distance} is None.")
        else:
            max_distance -= 0.001 # Small margin
            if max_distance < 0:
                return None

        # Add max distance to the candidate pose
        transformed_inserted_pose = np.eye(4)
        transformed_inserted_pose[2, 3] += max_distance

        # Push the gripper to the point cloud using maximum movable distance
        transformed_points.transform(np.linalg.inv(transformed_inserted_pose))

        # show debug
        if show_debug:
            o3d.visualization.draw_geometries([transformed_points]+self.meshes, window_name="Transformed Points")
        
        # Check 1: If the gripper's closure volume contains any points
        if not self._has_point_in_closure_volume(transformed_points):
            if show_debug:
                print("No points in closure volume.")
            return None

        # Check 2: If the gripper does not collide with given points
        if self._has_collision(transformed_points):
            if show_debug:
                print("Collision detected with the gripper.")
            return None

        inserted_candidate_pose = candidate_pose.copy()
        inserted_candidate_pose[2, 3] += max_distance
        return inserted_candidate_pose

    @property
    def meshes(self):
        """
        Get the list of meshes for the gripper components.

        Returns
        -------
        list
            A list of meshes for the gripper components.
        """
        return list(self._meshes.values())


def test1():
    gripper = Gripper([0.1, 0.03, 0.05], 0.01)
    meshes = gripper.meshes
    o3d.visualization.draw_geometries(meshes)

    import pickle as pkl
    with open('../data/darboux_frames/candidate_poses.pkl', 'rb') as f:
        candidate_poses = pkl.load(f)
    
    pcd = o3d.io.read_point_cloud('../data/segmented/remote_controller.ply')
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

    for pose in candidate_poses:
        gripper_pose = gripper.find_graspable_pose(pose, pcd, show_debug=True)
        if gripper_pose is not None:
            print(f'Original pose:\n{pose}')
            print(f'Graspable pose found:\n{gripper_pose}')
            print('Congratulations! Graspable pose found.')
            break
        else:
            print("No graspable pose found for the given candidate pose.")

if __name__ == "__main__":
    test1()