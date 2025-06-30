import abc

import mujoco
import numpy as np

from ..prelude import *

from ..environment import (
    add_geom,
    setup_option, setup_visual, setup_textures, add_nest, add_wall, add_robot, add_food_object,
    rand_robot_pos, rand_food_pos
)


def _generate_mjspec(settings: Settings) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    setup_visual(spec, settings)
    setup_textures(spec, settings)

    add_wall(spec, settings)

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
        rgba=GROUND_COLOR
    )

    add_nest(spec, settings)

    invalid_area: list[tuple[Position, float]] = []

    # Create food objects
    for i in range(settings.Food.NUM):
        if i < len(settings.Food.INITIAL_POSITION):
            position: Position = settings.Food.INITIAL_POSITION[i]
        else:
            position: Position = rand_food_pos(settings, invalid_area)
        invalid_area.append(
            (position, settings.Food.RADIUS)
        )
        add_food_object(spec, settings, i, position)

    # Create robots
    robot_specs = []
    for i in range(settings.Robot.NUM):
        if i < len(settings.Robot.INITIAL_POSITION):
            position: RobotLocation = settings.Robot.INITIAL_POSITION[i]
        else:
            position: RobotLocation = rand_robot_pos(settings, invalid_area)
        invalid_area.append(
            (position.position, settings.Robot.RADIUS)
        )
        robot_specs.append(
            add_robot(spec, settings, i, position)
        )

    return spec, robot_specs


class MujocoBackend(SimulatorBackend, abc.ABC):
    def __init__(self, settings: Settings, render: bool = False):
        self.settings = settings
        self.do_render = render
        self.render_shape = self.settings.Render.RENDER_WIDTH, self.settings.Render.RENDER_HEIGHT

        self.spec, robot_specs = _generate_mjspec(self.settings)
        self.model: mujoco.MjModel = self.spec.compile()
        # self.model: mujoco.MjModel = mujoco.MjModel.from_xml_string(self.spec.to_xml())
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self.robot_values = [RobotValues(self.data, s) for s in robot_specs]

        # Initialize renderer if needed
        self.camera = mujoco.MjvCamera()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return

        try:
            pos = np.array(pos)
            lookat = np.array(lookat)
            sub = pos - lookat
            self.camera.lookat[:] = lookat
            self.camera.distance = np.linalg.norm(sub)
            self.camera.azimuth = np.arctan2(
                sub[1], sub[0]
            ) * 180 / mujoco.mjPI
            self.camera.elevation = -np.arcsin(
                sub[2] / self.camera.distance
            ) * 180 / mujoco.mjPI
            # self.camera.elevation = -35

            if self.do_render:
                with mujoco.Renderer(self.model, width=self.render_shape[0], height=self.render_shape[1]) as renderer:
                    renderer.update_scene(self.data, self.camera)
                    renderer.render(out=img_buf)

        except Exception as e:
            print(f"MuJoCo render error: {e}")
            img_buf.fill(0)
