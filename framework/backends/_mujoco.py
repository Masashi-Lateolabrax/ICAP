import abc

import mujoco
import numpy as np

from icecream import ic
from ..prelude import *

from ..environment import (
    add_geom,
    add_texture, add_material,
    add_nest, add_wall, add_robot, add_food_object,
    rand_robot_pos, rand_food_pos
)


class MujocoBackend(SimulatorBackend, abc.ABC):
    @staticmethod
    def _generate_base_mjspec(
            settings: Settings
    ) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()

        # Simulation settings
        spec.option.timestep = settings.Simulation.TIME_STEP
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        # Visual settings
        visual: mujoco._specs.MjVisual = spec.visual
        visual.global_.offwidth = settings.Render.RENDER_WIDTH
        visual.global_.offheight = settings.Render.RENDER_HEIGHT

        # Texture and material setup
        add_texture(
            spec,
            name="simple_checker",
            type_=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=CHECKER_TEXTURE_SIZE,
            height=CHECKER_TEXTURE_SIZE,
            rgb1=CHECKER_RGB_WHITE,
            rgb2=CHECKER_RGB_GRAY
        )
        add_material(
            spec,
            name="ground",
            texture="simple_checker",
            texrepeat=(
                settings.Simulation.WORLD_WIDTH * 0.5,
                settings.Simulation.WORLD_HEIGHT * 0.5
            )
        )

        return spec

    def __init__(self, settings: Settings, render: bool = False):
        self.settings = settings
        self.do_render = render
        self.render_shape = self.settings.Render.RENDER_WIDTH, self.settings.Render.RENDER_HEIGHT

        self.spec = self._generate_base_mjspec(self.settings)

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
                add_food_object(self.spec, settings, i, position)
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
                add_robot(self.spec, settings, i, position)
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
            ) * 180 / mujoco.mjPI + 180
            self.camera.elevation = -np.arcsin(
                sub[2] / self.camera.distance
            ) * 180 / mujoco.mjPI
            # self.camera.elevation = -35

            if self.do_render:
                with mujoco.Renderer(self.model, width=self.render_shape[0], height=self.render_shape[1]) as renderer:
                    renderer.update_scene(self.data, self.camera)
                    renderer.render(out=img_buf)

        except Exception as e:
            ic("MuJoCo render error:", e)
            img_buf.fill(0)
