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
        max_range: float
    ):
        """
        DepthSensor constructor.

        Args:
            robot (RobotValues): Robot state values for position and orientation
            model (mujoco.MjModel): MuJoCo model for raycasting
            data (mujoco.MjData): MuJoCo data for current state
            num_rays (int): Number of rays to cast (evenly distributed 360 degrees)
            max_range (float): Maximum detection range for rays
        """
        self.robot = robot
        self.model = model
        self.data = data
        self.num_rays = num_rays
        self.max_range = max_range

        # Get robot's body ID from its site
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, robot.site.name)
        self.robot_body_id = model.site_bodyid[site_id]

        # Pre-compute ray angles (evenly distributed in 360 degrees)
        self.ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        # Pre-allocate arrays for performance
        self.distances = np.ones(num_rays, dtype=np.float32) * max_range
        self.ray_origins = np.zeros((num_rays, 3), dtype=np.float64)
        self.ray_directions = np.zeros((num_rays, 3), dtype=np.float64)
        self.geomids = np.full(num_rays, -1, dtype=np.int32)

    def get(self) -> np.ndarray:
        """
        Cast rays and return distances to nearest objects.

        Returns:
            np.ndarray: Array of distances (normalized to [0, 1] range)
                       Shape: (num_rays,)
        """
        # Get robot position and orientation
        robot_pos = self.robot.xpos
        robot_direction = self.robot.xdirection

        # Calculate robot's yaw angle from direction vector
        robot_yaw = np.arctan2(robot_direction[1], robot_direction[0])

        # Prepare all ray origins (same position for all rays)
        self.ray_origins[:, 0] = robot_pos[0]
        self.ray_origins[:, 1] = robot_pos[1]
        self.ray_origins[:, 2] = 0.05  # Slightly above ground to avoid floor collision

        # Calculate all ray directions at once
        absolute_angles = robot_yaw + self.ray_angles
        self.ray_directions[:, 0] = np.cos(absolute_angles)
        self.ray_directions[:, 1] = np.sin(absolute_angles)
        self.ray_directions[:, 2] = 0.0

        # Cast all rays at once using mj_multiRay
        mujoco.mj_multiRay(
            self.model,
            self.data,
            self.ray_origins.flatten(),
            self.ray_directions.flatten(),
            geomgroup=None,
            flg_static=1,
            bodyexclude=self.robot_body_id,
            geomid=self.geomids,
            dist=self.distances,
            nray=self.num_rays,
            cutoff=self.max_range
        )

        # mj_multiRay returns -1 for no hit, replace with max_range
        self.distances[self.distances < 0] = self.max_range

        # Normalize distances to [0, 1] range (0=far, 1=close)
        normalized_distances = 1.0 - (self.distances / self.max_range)

        return normalized_distances