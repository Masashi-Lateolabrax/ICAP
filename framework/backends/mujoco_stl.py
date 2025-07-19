import os.path

import mujoco

from ..prelude import *

from ._mujoco import MujocoBackend

from ..environment import (
    add_mesh_in_asset, MeshContentType,
    add_geom,
    add_nest, add_wall,
    add_food_object_with_mesh, add_robot_with_mesh,
    rand_robot_pos, rand_food_pos,
)


class MujocoSTL(MujocoBackend):
    def __init__(self, settings: Settings, render: bool = False):
        self.settings = settings
        self.do_render = render
        self.render_shape = self.settings.Render.RENDER_WIDTH, self.settings.Render.RENDER_HEIGHT

        self.spec = self._generate_base_mjspec(self.settings)

        # Add STL mesh assets
        food_mesh = add_mesh_in_asset(
            self.spec,
            name="food_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "food-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        robot_mesh = add_mesh_in_asset(
            self.spec,
            name="robot_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "robot-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        add_wall(self.spec, settings)

        add_geom(
            self.spec.worldbody,
            geom_type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=(0, 0, 0),
            size=(
                settings.Simulation.WORLD_WIDTH * 0.5,
                settings.Simulation.WORLD_HEIGHT * 0.5,
                1
            ),
            material="ground",
            rgba=GROUND_COLOR,
            condim=GROUND_COLLISION_CONDIM
        )

        nest_spec = add_nest(self.spec, settings)

        invalid_area: list[tuple[Position, float]] = []

        # Create food objects
        food_specs = []
        for i in range(settings.Food.NUM):
            if i < len(settings.Food.INITIAL_POSITION):
                position: Position = settings.Food.INITIAL_POSITION[i]
            else:
                position: Position = rand_food_pos(
                    settings.Simulation.WORLD_WIDTH, settings.Simulation.WORLD_HEIGHT,
                    settings.Food.RADIUS,
                    invalid_area
                )
            invalid_area.append(
                (position, settings.Food.RADIUS)
            )
            food_specs.append(
                add_food_object_with_mesh(self.spec, settings, food_mesh, i, position)
            )

        # Create robots
        robot_specs = []
        for i in range(settings.Robot.NUM):
            if i < len(settings.Robot.INITIAL_POSITION):
                position: RobotLocation = settings.Robot.INITIAL_POSITION[i]
            else:
                position: RobotLocation = rand_robot_pos(
                    settings.Simulation.WORLD_WIDTH, settings.Simulation.WORLD_HEIGHT,
                    settings.Robot.RADIUS,
                    invalid_area
                )
            invalid_area.append(
                (position.position, settings.Robot.RADIUS)
            )
            robot_specs.append(
                add_robot_with_mesh(self.spec, settings, robot_mesh, i, position)
            )

        # Instantiate the MuJoCo model and data
        self.model: mujoco.MjModel = self.spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # Extract important value references
        self.nest_site = self.data.site(nest_spec.name)
        self.robot_values = [
            RobotValues(settings.Robot.DISTANCE_BETWEEN_WHEELS, settings.Robot.MAX_SPEED, self.data, s)
            for s in robot_specs
        ]
        self.food_values = [FoodValues(self.data, s) for s in food_specs]

        # Miscellaneous initializations
        self.camera = mujoco.MjvCamera()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_scores(self) -> list[float]:
        return [0.0]

    def calc_total_score(self) -> float:
        return 0.0
