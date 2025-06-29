import mujoco

from ..config.settings import Settings
from ..interfaces import SimulatorBackend
from ..mujoco_utils import create_environment


class MujocoBackend(SimulatorBackend):
    def __init__(self, settings: Settings, render: bool = False):
        self.settings = settings
        self.xml = create_environment(settings).to_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)

        # Initialize renderer if needed
        self.renderer = None
        if render:
            self.renderer = mujoco.Renderer(self.model)
            self.renderer.scene.camera.lookat[:] = [0, 0, 0]
            self.renderer.scene.camera.distance = 15
            self.renderer.scene.camera.elevation = -30

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.renderer is not None:
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def get_robot_position(self, robot_id: int):
        robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"robot{robot_id}")
        if robot_body_id == -1:
            return None
        return self.data.xpos[robot_body_id].copy()

    def set_robot_control(self, robot_id: int, vx: float, vy: float, omega: float):
        actuator_names = [
            f"robot{robot_id}_slide_x_act",
            f"robot{robot_id}_slide_y_act",
            f"robot{robot_id}_hinge_act"
        ]

        controls = [vx, vy, omega]

        for name, control in zip(actuator_names, controls):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id != -1:
                self.data.ctrl[actuator_id] = control


def example_run():
    settings = Settings()
    backend = MujocoBackend(settings)
    print(backend.xml)
    # for i in range(1000):
    #     backend.step()
