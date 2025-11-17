import numpy as np
import mujoco

from ..prelude import *


class DepthSensor(SensorInterface):
    """
    Ray-based depth sensor using MuJoCo's raycasting capabilities.

    Casts rays in a circular pattern around the robot to detect distances to objects.
    Returns an array of distances for each ray direction.
    """

    def __init__(
        self,
        robot: RobotValues,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        num_rays: int,
        max_range: float,
        offset: float = 0.0
    ):
        """
        DepthSensor constructor.

        Args:
            robot (RobotValues): Robot state values for position and orientation
            model (mujoco.MjModel): MuJoCo model for raycasting
            data (mujoco.MjData): MuJoCo data for current state
            num_rays (int): Number of rays to cast (evenly distributed 360 degrees)
            max_range (float): Maximum detection range for rays
            offset (float): Radial offset from robot center (e.g., robot radius)
        """
        self.robot = robot
        self.model = model
        self.data = data
        self.num_rays = num_rays
        self.max_range = max_range
        self.offset = offset

        # Get robot's body ID from its site
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.site.name)
        self.robot_body_id = model.site_bodyid[site_id]

        # Pre-compute ray angles (evenly distributed in 360 degrees)
        self.ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        # Pre-allocate arrays for performance
        # Python bindings require column vectors [n, 1] for geomid and dist
        self.distances = np.ones((num_rays, 1), dtype=np.float64) * max_range
        self.ray_origins = np.zeros((num_rays, 3), dtype=np.float64)
        self.ray_directions = np.zeros((num_rays, 3), dtype=np.float64)
        self.geomids = np.full((num_rays, 1), -1, dtype=np.int32)

    def get(self) -> np.ndarray:
        """
        Cast rays and return distances to nearest objects.

        Returns:
            np.ndarray: Array of distances (normalized to [0, 1] range)
                       Shape: (num_rays,)
                       1.0 = close (distance 0), 0.0 = far (distance max_range)
        """
        # Get robot position and orientation
        robot_pos = self.robot.xpos
        robot_direction = self.robot.xdirection

        # Calculate robot's yaw angle from direction vector
        robot_yaw = np.arctan2(robot_direction[1], robot_direction[0])

        # Single ray origin (all rays emanate from robot center)
        # Python bindings expect column vector [3, 1]
        ray_origin = np.array([[robot_pos[0]], [robot_pos[1]], [0.05]], dtype=np.float64)

        # Calculate all ray directions at once
        absolute_angles = robot_yaw + self.ray_angles
        self.ray_directions[:, 0] = np.cos(absolute_angles)
        self.ray_directions[:, 1] = np.sin(absolute_angles)
        self.ray_directions[:, 2] = 0.0

        # Python bindings expect vec as column vector [nray*3, 1]
        vec = self.ray_directions.flatten().reshape(-1, 1)

        # Cast all rays at once using mj_multiRay
        # cutoff includes offset to detect objects within max_range from robot surface
        mujoco.mj_multiRay(
            self.model,
            self.data,
            ray_origin,                      # Single origin point [3, 1]
            vec,                              # Ray directions [num_rays*3, 1]
            geomgroup=None,
            flg_static=1,
            bodyexclude=self.robot_body_id,
            geomid=self.geomids,
            dist=self.distances,
            nray=self.num_rays,
            cutoff=self.max_range + self.offset
        )

        # mj_multiRay returns -1 for no hit, replace with max_range + offset
        self.distances[self.distances < 0] = self.max_range + self.offset

        # Subtract offset to get distance from robot surface
        self.distances -= self.offset
        
        # Clamp to [0, max_range]
        np.clip(self.distances, 0.0, self.max_range, out=self.distances)

        # Normalize distances to [0, 1] range (1=close, 0=far)
        # Flatten from (num_rays, 1) to (num_rays,) for output
        normalized_distances = 1.0 - (self.distances.flatten() / self.max_range)

        return normalized_distances