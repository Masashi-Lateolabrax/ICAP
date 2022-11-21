import mujoco
import numpy

from studyLib import optimizer, wrap_mjc, miscellaneous
from studyLib.optimizer import EnvInterface, MuJoCoEnvInterface


def _gen_env(
        robot_pos: (float, float),
        obstacle_pos: (float, float),
        obstacle_weight: float
):
    generator = wrap_mjc.MuJoCoXMLGenerator("co-behavior")

    generator.add_option({
        "timestep": 0.033333,
        "gravity": "0 0 -981.0"
    })
    generator.add_asset().add_texture({
        "type": "skybox",
        "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7",
        "rgb2": "0 0 0",
        "width": "512",
        "height": "512",
    })

    worldbody = generator.get_body()
    act = generator.add_actuator()

    # Create Ground
    worldbody.add_geom({
        "type": "plane",
        "size": "0 0 0.05",
        "rgba": "1.0 1.0 1.0 0.5",
        "condim": "1",
        "contype": "1",
        "conaffinity": "2",
        "priority": "0"
    })

    # Create Obstacle
    obstacle_body = worldbody.add_body({
        "name": "obstacle",
        "pos": f"{obstacle_pos[0]} {obstacle_pos[1]} 20"
    })
    obstacle_body.add_site({"name": "s1", "pos": "0 100 0"})
    obstacle_body.add_freejoint()
    obstacle_body.add_geom({
        "name": "geom_obstacle",
        "type": "cylinder",
        "size": "100 10",
        "mass": f"{obstacle_weight}",
        "rgba": "1 0 0 1",
        "condim": "3",
    })

    # Create Robots
    depth = 1.0
    friction = (1.0, 0.005, 0.0001)  # default : (1.0, 0.005, 0.0001)
    body_density = 0.51995  # 鉄の密度(7.874 g/cm^3), ルンバの密度(0.51995 g/cm^3)
    wheel_density = 0.3

    robot_body = worldbody.add_body({
        "name": f"robot",
        "pos": f"{robot_pos[0]} {robot_pos[1]} {10 + depth + 0.5}",
        "axisangle": f"0 0 1 0",
    })
    robot_body.add_freejoint()
    robot_body.add_site({"name": "s2", "pos": "0 -17.5 0"})
    robot_body.add_geom({
        "type": "cylinder",
        "size": "17.5 5",  # 幅35cm，高さ10cm
        "density": f"{body_density}",
        "rgba": "1 1 0 0.3",
        "condim": "1",
        "contype": "2",
        "conaffinity": "2"
    })

    right_wheel_body = robot_body.add_body({"pos": f"10 0 -{depth}"})
    right_wheel_body.add_joint({"name": f"joint_robot_right", "type": "hinge", "axis": "-1 0 0"})
    right_wheel_body.add_geom({
        "type": "cylinder",
        "size": "5 5",
        "density": f"{wheel_density}",
        "axisangle": "0 1 0 90",
        "condim": "6",
        "contype": "1",
        "conaffinity": "1",
        "priority": "1",
        "friction": f"{friction[0]} {friction[1]} {friction[2]}"
    })

    left_wheel_body = robot_body.add_body({"pos": f"-10 0 -{depth}"})
    left_wheel_body.add_joint({"name": f"joint_robot_left", "type": "hinge", "axis": "-1 0 0"})
    left_wheel_body.add_geom({
        "type": "cylinder",
        "size": "5 5",
        "density": f"{wheel_density}",
        "axisangle": "0 1 0 90",
        "condim": "6",
        "contype": "1",
        "conaffinity": "1",
        "priority": "1",
        "friction": f"{friction[0]} {friction[1]} {friction[2]}"
    })

    front_wheel_body = robot_body.add_body({"pos": f"0 15 {-5 + 1.5 - depth}"})
    front_wheel_body.add_joint({"type": "ball"})
    front_wheel_body.add_geom({
        "type": "sphere",
        "size": "1.5",
        "density": f"{wheel_density}",
        "condim": "1",
        "contype": "1",
        "conaffinity": "1",
        "priority": "1"
    })

    rear_wheel_body = robot_body.add_body({"pos": f"0 -15 {-5 + 1.5 - depth}"})
    rear_wheel_body.add_joint({"type": "ball"})
    rear_wheel_body.add_geom({
        "type": "sphere",
        "size": "1.5",
        "density": f"{wheel_density}",
        "condim": "1",
        "contype": "1",
        "conaffinity": "1",
        "priority": "1"
    })

    act.add_velocity({
        "name": f"act_robot_left",
        "joint": f"joint_robot_left",
        "gear": "80",
        "ctrllimited": "true",
        "ctrlrange": "-50000 50000",
        "kv": "1"
    })
    act.add_velocity({
        "name": f"act_robot_right",
        "joint": f"joint_robot_right",
        "gear": "80",
        "ctrllimited": "true",
        "ctrlrange": "-50000 50000",
        "kv": "1"
    })

    tendon = generator.add_tendon()
    tendon_spatial = tendon.add_spatial({
        "width": "5",
        "limited": "true",
        "range": "0 120",
        "rgba": "0 1 0 1"
    })
    tendon_spatial.add_point({"site": "s1"})
    tendon_spatial.add_point({"site": "s2"})

    return generator.generate()


class _Obstacle:
    def __init__(self, model: wrap_mjc.WrappedModel):
        self.body = model.get_body(f"obstacle")

    def get_pos(self):
        return self.body.get_xpos().copy()


class _Robot:
    def __init__(self, model: wrap_mjc.WrappedModel):
        self.body = model.get_body(f"robot")
        self.left_act = model.get_act(f"act_robot_left")
        self.right_act = model.get_act(f"act_robot_right")

    def get_pos(self) -> numpy.ndarray:
        return self.body.get_xpos().copy()

    def get_orientation(self):
        rot_mat = numpy.zeros(9).reshape(3, 3)
        mujoco.mju_quat2Mat(rot_mat.ravel(), self.body.get_xquat())
        return rot_mat

    def get_direction(self):
        return numpy.dot(self.get_orientation(), [0.0, 1.0, 0.0])

    def act(self):
        self.left_act.ctrl = 10000
        self.right_act.ctrl = 10000


def _evaluate(
        robot_pos: (float, float),
        obstacle_pos: (float, float),
        obstacle_weight: float,
        timestep: int,
        camera: wrap_mjc.Camera = None,
        window: miscellaneous.Window = None
) -> float:
    xml = _gen_env(robot_pos, obstacle_pos, obstacle_weight)
    model = wrap_mjc.WrappedModel(xml)

    if not (camera is None):
        model.set_camera(camera)

    obstacle = _Obstacle(model)
    robot = _Robot(model)

    for _ in range(0, 30):
        model.step()

    for t in range(0, timestep):
        # Simulate
        robot.act()
        model.step()

        # Render MuJoCo Scene
        if window is not None:
            if not window.render(model, camera):
                exit()
            window.flush()

    return numpy.linalg.norm(obstacle.get_pos(), ord=2)


class Environment(optimizer.MuJoCoEnvInterface):
    def __init__(
            self,
            robot_pos: (float, float),
            obstacle_pos: (float, float),
            timestep: int
    ):
        self.robot_pos = robot_pos
        self.obstacle_pos = obstacle_pos
        self.timestep = timestep

    def calc(self, para) -> float:
        return _evaluate(
            self.robot_pos,
            self.obstacle_pos,
            para,
            self.timestep
        )

    def calc_and_show(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> float:
        return _evaluate(
            self.robot_pos,
            self.obstacle_pos,
            para,
            self.timestep,
            camera,
            window
        )


class EnvCreator(optimizer.MuJoCoEnvCreator):
    """
    ロボットが前方にしか進めない状態で，後方の敵を押し巣から遠くに運ぶ環境．
    ロボットはy軸の正の方向を正面として設置される．
    """

    def __init__(self):
        self.obstacle_pos: (float, float) = (0, 0)
        self.robot_pos: (float, float) = (0, 0)
        self.timestep: int = 100

    def save(self):
        import struct

        packed = struct.pack(
            "ddddI",
            self.obstacle_pos[0],
            self.obstacle_pos[1],
            self.robot_pos[0],
            self.robot_pos[1],
            self.timestep
        )
        return packed

    def load(self, data: bytes, offset: int = 0) -> int:
        import struct

        # 障害物の座標
        s = offset
        e = 16
        self.obstacle_pos = struct.unpack("<dd", data[s:e])[0:2]

        # ロボットの座標
        s = offset
        e = 16
        self.obstacle_pos = struct.unpack("<dd", data[s:e])[0:2]

        # タイムステップ
        s = offset
        e = 4
        self.timestep = struct.unpack("<I", data[s:e])[0]

        return e - offset

    def dim(self) -> int:
        return 1

    def create(self) -> EnvInterface:
        return Environment(
            self.robot_pos,
            self.obstacle_pos,
            self.timestep
        )

    def create_mujoco_env(self) -> MuJoCoEnvInterface:
        return Environment(
            self.robot_pos,
            self.obstacle_pos,
            self.timestep
        )
