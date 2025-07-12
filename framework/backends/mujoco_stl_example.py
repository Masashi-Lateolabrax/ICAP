import os.path

import mujoco
import numpy as np

from icecream import ic
from ..prelude import *

from ._mujoco import MujocoBackend

from ..environment import (
    add_mesh_in_asset, add_geom, MeshContentType,
    setup_option, setup_visual, setup_textures, add_nest, rand_food_pos,
    add_food_object_with_mesh, add_robot_with_mesh
)


def _generate_mjspec(
        settings: Settings
) -> tuple[mujoco.MjSpec, mujoco._specs.MjsSite, RobotSpec, FoodSpec]:
    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    setup_visual(spec, settings)
    setup_textures(spec, settings)

    food_mesh = add_mesh_in_asset(
        spec,
        name="food_mesh",
        file=os.path.abspath("../assets/food-object.stl"),
        content_type=MeshContentType.STL,
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    )

    robot_mesh = add_mesh_in_asset(
        spec,
        name="robot_mesh",
        file=os.path.abspath("../assets/robot-object.stl"),
        content_type=MeshContentType.STL,
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    )

    add_geom(
        spec.worldbody,
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

    nest_spec = add_nest(spec, settings)

    invalid_area: list[tuple[Position, float]] = []

    if len(settings.Food.INITIAL_POSITION) > 0:
        position: Position = settings.Food.INITIAL_POSITION[0]
    else:
        position: Position = rand_food_pos(settings, invalid_area)
    invalid_area.append(
        (position, settings.Food.RADIUS)
    )
    food_spec = add_food_object_with_mesh(spec, settings, food_mesh, 0, position)

    # Create robots
    position: RobotLocation = settings.Robot.INITIAL_POSITION[0]
    invalid_area.append(
        (position.position, settings.Robot.RADIUS)
    )
    robot_spec = add_robot_with_mesh(spec, settings, robot_mesh, 0, position)

    return spec, nest_spec, robot_spec, food_spec


class ExampleSTL(MujocoBackend):
    def __init__(self, settings: Settings, render: bool = False):
        self.settings = settings
        self.do_render = render
        self.render_shape = self.settings.Render.RENDER_WIDTH, self.settings.Render.RENDER_HEIGHT

        self.spec, nest_spec, robot_spec, food_spec = _generate_mjspec(self.settings)

        self.model: mujoco.MjModel = self.spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self.nest_site = self.data.site(nest_spec.name)

        self.camera = mujoco.MjvCamera()

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def score(self) -> list[float]:
        return [0.0]

    def total_score(self) -> float:
        return 0.0
