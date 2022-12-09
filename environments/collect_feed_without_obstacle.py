import copy
from typing import Sequence

import mujoco
import numpy

from studyLib import optimizer, wrap_mjc, miscellaneous, nn_tools
from studyLib.nn_tools import la


def _gen_env(
        nest_pos: (float, float),
        robot_pos: list[(float, float, float)],
        feed_pos: list[(float, float)],
        pheromone_field_panel_size: float,
        pheromone_field_pos: (float, float),
        pheromone_field_shape: (int, int)
):
    generator = wrap_mjc.MuJoCoXMLGenerator("co-behavior")

    generator.add_option({
        "timestep": 0.033333,
        "gravity": "0 0 -981.0",
        "impratio": "3",
    })

    ######################################################################################################
    # Set Texture
    ######################################################################################################
    asset = generator.add_asset()
    asset.add_texture({
        "type": "skybox",
        "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7",
        "rgb2": "0 0 0",
        "width": "512",
        "height": "512",
    })
    asset.add_texture({
        "name": "wheel_texture",
        "type": "cube",
        "builtin": "checker",
        "width": "16",
        "height": "16",
    })
    asset.add_material({
        "name": "wheel_material",
        "texture": "wheel_texture",
    })

    ######################################################################################################
    # Default Setting
    ######################################################################################################
    default = generator.add_default()
    default.add_geom({
        "density": "1",  # Unit is g/cm^3. (1000kg/m^3 = 1g/cm^3)
        "solimp": "0.9 0.95 1.0 0.5 2",
        "solref": "0.002 100.0",
    })

    ######################################################################################################
    # Create Setting for Walls
    ######################################################################################################
    wall_height = 15.0
    wall_default = default.add_default("wall")
    wall_default.add_geom({
        "type": "box",
        "rgba": "0.7 0.7 0.7 0.5",
        "condim": "1",
        "priority": "2",
        "contype": "1",
        "conaffinity": "2",
    })

    ######################################################################################################
    # Create Setting for Feeds
    ######################################################################################################
    feed_default = default.add_default("feeds")
    feed_default.add_geom({
        "type": "cylinder",
        "size": "50 10",
        "mass": "3000",  # 3kg = 3000g
        "rgba": "0 1 1 1",
        "condim": "3",
        "priority": "1",
        "friction": "0.5 0.0 0.0",
        "contype": "2",
        "conaffinity": "2",
    })

    ######################################################################################################
    # Create Setting for Robots
    ######################################################################################################
    body_default = default.add_default("robot_body")
    body_default.add_geom({
        "type": "cylinder",
        "size": "17.5 5",  # 幅35cm，高さ10cm
        "mass": "3000",  # 3kg = 3000g．ルンバの重さが4kgくらい．
        "rgba": "1 1 0 0.3",
        "condim": "1",
        "priority": "0",
        "contype": "2",
        "conaffinity": "2"
    })
    wheel_default = default.add_default("robot_wheel")
    wheel_default.add_geom({
        "type": "cylinder",
        "size": "5 5",
        "mass": "250",
        "axisangle": "0 1 0 90",
        "condim": "6",
        "priority": "1",
        "friction": "0.2 0.01 0.01",
        "contype": "1",
        "conaffinity": "1",
        "material": "wheel_material"
    })
    ball_default = wheel_default.add_default("robot_ball")
    ball_default.add_geom({
        "type": "sphere",
        "size": "1.5",
        "condim": "1",
    })

    worldbody = generator.get_body()
    act = generator.add_actuator()
    sensor = generator.add_sensor()

    ######################################################################################################
    # Create Ground
    ######################################################################################################
    worldbody.add_geom({
        "type": "plane",
        "size": "0 0 0.05",
        "rgba": "1.0 1.0 1.0 0.0",
        "condim": "1",
        "priority": "0",
        "contype": "1",
        "conaffinity": "2",
    })

    ######################################################################################################
    # Create Wall
    ######################################################################################################
    big_wall_size = (
        pheromone_field_panel_size * (pheromone_field_shape[0] + 2),
        pheromone_field_panel_size * (pheromone_field_shape[1] + 2)
    )
    wall_size = (
        pheromone_field_panel_size * pheromone_field_shape[0],
        pheromone_field_panel_size * pheromone_field_shape[1]
    )
    worldbody.add_geom({
        "class": "wall",
        "pos": f"{pheromone_field_pos[0] + 0.5 * (wall_size[0] + pheromone_field_panel_size)} {pheromone_field_pos[1]} {wall_height * 0.5}",
        "size": f"{pheromone_field_panel_size} {big_wall_size[1] * 0.5} {wall_height * 0.5}",
    })
    worldbody.add_geom({
        "class": "wall",
        "pos": f"{pheromone_field_pos[0] - 0.5 * (wall_size[0] + pheromone_field_panel_size)} {pheromone_field_pos[1]} {wall_height * 0.5}",
        "size": f"{pheromone_field_panel_size} {big_wall_size[1] * 0.5} {wall_height * 0.5}",
    })
    worldbody.add_geom({
        "class": "wall",
        "pos": f"{pheromone_field_pos[0]} {pheromone_field_pos[1] + 0.5 * (wall_size[1] + pheromone_field_panel_size)} {wall_height * 0.5}",
        "size": f"{wall_size[0] * 0.5} {pheromone_field_panel_size} {wall_height * 0.5}",
    })
    worldbody.add_geom({
        "class": "wall",
        "pos": f"{pheromone_field_pos[0]} {pheromone_field_pos[1] - 0.5 * (wall_size[1] + pheromone_field_panel_size)} {wall_height * 0.5}",
        "size": f"{wall_size[0] * 0.5} {pheromone_field_panel_size} {wall_height * 0.5}",
    })

    ######################################################################################################
    # Create Nest
    ######################################################################################################
    worldbody.add_geom({
        "type": "cylinder",
        "pos": f"{nest_pos[0]} {nest_pos[1]} -3",
        "size": "150 1",
        "rgba": "0.0 1.0 0.0 1",
    })

    ######################################################################################################
    # Create Feeds
    ######################################################################################################
    for i, fp in enumerate(feed_pos):
        feed_body = worldbody.add_body({
            "name": f"feed{i}",
            "pos": f"{fp[0]} {fp[1]} 11"
        })
        feed_body.add_freejoint()
        feed_body.add_site({"name": f"site_feed{i}"})
        feed_body.add_geom({"class": "feeds"})
        sensor.add_velocimeter({
            "name": f"sensor_feed{i}_velocity",
            "site": f"site_feed{i}"
        })

    ######################################################################################################
    # Create Robots
    ######################################################################################################
    depth = 1.0
    for i, rp in enumerate(robot_pos):
        robot_body = worldbody.add_body({
            "name": f"robot{i}",
            "pos": f"{rp[0]} {rp[1]} {10 + depth + 0.5}",
            "axisangle": f"0 0 1 {rp[2]}",
        })
        robot_body.add_freejoint()
        robot_body.add_geom({"class": "robot_body"})

        right_wheel_body = robot_body.add_body({"pos": f"10 0 -{depth}"})
        right_wheel_body.add_joint({"name": f"joint_robot{i}_right", "type": "hinge", "axis": "-1 0 0"})
        right_wheel_body.add_geom({"class": "robot_wheel"})

        left_wheel_body = robot_body.add_body({"pos": f"-10 0 -{depth}"})
        left_wheel_body.add_joint({"name": f"joint_robot{i}_left", "type": "hinge", "axis": "-1 0 0"})
        left_wheel_body.add_geom({"class": "robot_wheel"})

        front_wheel_body = robot_body.add_body({"pos": f"0 15 {-5 + 1.5 - depth}"})
        front_wheel_body.add_joint({"type": "ball"})
        front_wheel_body.add_geom({"class": "robot_ball"})

        rear_wheel_body = robot_body.add_body({"pos": f"0 -15 {-5 + 1.5 - depth}"})
        rear_wheel_body.add_joint({"type": "ball"})
        rear_wheel_body.add_geom({"class": "robot_ball"})

        act.add_velocity({
            "name": f"act_robot{i}_left",
            "joint": f"joint_robot{i}_left",
            "kv": "100",
            "gear": "30",
        })
        act.add_velocity({
            "name": f"act_robot{i}_right",
            "joint": f"joint_robot{i}_right",
            "kv": "100",
            "gear": "30"
        })

    ######################################################################################################
    # Generate XML
    ######################################################################################################
    xml = generator.generate()
    return xml


class _Feed:
    def __init__(self, model: wrap_mjc.WrappedModel, number: int):
        self._body = model.get_body(f"feed{number}")
        self._velocity_sensor = model.get_sensor(f"sensor_feed{number}_velocity")
        self._init_pos: numpy.ndarray = None

    def get_pos(self):
        return self._body.get_xpos().copy()

    def get_velocity(self) -> numpy.ndarray:
        return self._velocity_sensor.get_data()

    def get_init_pos(self) -> numpy.ndarray:
        if self._init_pos is None:
            self._init_pos = self._body.get_xpos().copy()
        return self._init_pos.copy()


class ConvertPheromone(nn_tools.interface.CalcActivator):
    def __init__(self, num_node: int, pheromone_index: int):
        super().__init__(num_node)
        self.pheromone_index = pheromone_index

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        la.copyto(output, input_)
        output[self.pheromone_index] = (output[self.pheromone_index] + 1.0) * 0.5
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        return None


class RobotBrain:
    def __init__(self, para):
        # DIM 330

        self._calculator = nn_tools.Calculator(7)
        self._calculator.add_layer(nn_tools.BufLayer(7))  # 0

        pheromone_calculator = nn_tools.Calculator(7)  # 1->0
        pheromone_calculator.add_layer(nn_tools.FilterLayer([6]))  # 1->0->0
        pheromone_calculator.add_layer(nn_tools.AffineLayer(3))  # 1->0->1
        pheromone_calculator.add_layer(nn_tools.TanhLayer(3))  # 1->0->2
        pheromone_calculator.add_layer(nn_tools.AffineLayer(5))  # 1->0->1
        pheromone_calculator.add_layer(nn_tools.TanhLayer(5))  # 1->0->2
        pheromone_calculator.add_layer(nn_tools.AffineLayer(10))  # 1->0->1
        pheromone_calculator.add_layer(nn_tools.SigmoidLayer(10))  # 1->0->2
        pheromone_calculator.add_layer(nn_tools.BufLayer(10))  # 1->0->3

        state_calculator = nn_tools.Calculator(7)  # 1->1
        state_calculator.add_layer(nn_tools.FilterLayer([0, 1, 2, 3, 4, 5]))  # 1->0->0
        state_calculator.add_layer(nn_tools.AffineLayer(12))  # 1->1->1
        state_calculator.add_layer(nn_tools.TanhLayer(12))  # 1->1->2
        state_calculator.add_layer(nn_tools.AffineLayer(10))  # 1->1->1
        state_calculator.add_layer(nn_tools.TanhLayer(10))  # 1->1->2
        state_calculator.add_layer(nn_tools.BufLayer(10))  # 1->1->3

        self._calculator.add_layer(nn_tools.ParallelLayer([  # 1
            pheromone_calculator,  # 1->0
            state_calculator  # 1->1
        ]))
        self._calculator.add_layer(nn_tools.MulFoldLayer(10))  # 2
        self._calculator.add_layer(nn_tools.IsMaxLayer(10))  # 3
        self._calculator.add_layer(nn_tools.BufLayer(10))  # 4
        self._calculator.add_layer(nn_tools.InnerDotLayer(3))  # 5
        self._calculator.add_layer(nn_tools.TanhLayer(3))  # 6
        self._calculator.add_layer(ConvertPheromone(3, 2))  # 7
        self._calculator.add_layer(nn_tools.BufLayer(3))  # 8

        if not (para is None):
            self._calculator.load(para)

    def num_dim(self):
        return self._calculator.num_dim()

    def calc(self, array):
        return self._calculator.calc(array)

    def get_input(self) -> numpy.ndarray:
        buf_layer: nn_tools.BufLayer = self._calculator.get_layer(0)
        return buf_layer.buf.copy()

    def get_calced_feature_value(self) -> numpy.ndarray:
        buf_layer: nn_tools.BufLayer = self._calculator.get_layer(4)
        return buf_layer.buf.copy()

    def get_action(self, i: int) -> numpy.ndarray:
        inner_dot_layer: nn_tools.InnerDotLayer = self._calculator.get_layer(5)
        return numpy.array(inner_dot_layer.weights[:, i])

    def get_mod_p(self) -> numpy.ndarray:
        para_layer: nn_tools.ParallelLayer = self._calculator.get_layer(1)
        pheromone_calc: nn_tools.Calculator = para_layer.calcs[0]
        buf_layer: nn_tools.BufLayer = pheromone_calc.get_layer(2)
        return buf_layer.buf.copy()

    def get_mod_s(self) -> numpy.ndarray:
        para_layer: nn_tools.ParallelLayer = self._calculator.get_layer(1)
        pheromone_calc: nn_tools.Calculator = para_layer.calcs[1]
        buf_layer: nn_tools.BufLayer = pheromone_calc.get_layer(2)
        return buf_layer.buf.copy()

    def get_output(self) -> numpy.ndarray:
        buf_layer: nn_tools.BufLayer = self._calculator.get_layer(8)
        return buf_layer.buf.copy()


class _Robot:
    def __init__(self, brain: RobotBrain, model: wrap_mjc.WrappedModel, number: int):
        self.brain = brain
        self._body = model.get_body(f"robot{number}")
        self._left_act = model.get_act(f"act_robot{number}_left")
        self._right_act = model.get_act(f"act_robot{number}_right")

        # self._prev_action = 31
        # pos = self._body.get_xpos()
        # self._state_ball = model.add_deco_geom(mujoco.mjtGeom.mjGEOM_SPHERE)
        # self._state_ball.set_size([20, 20, 20])
        # self._state_ball.set_pos([pos[0], pos[1], 5])

    def get_pos(self) -> numpy.ndarray:
        return self._body.get_xpos().copy()

    def get_orientation(self) -> numpy.ndarray:
        rot_mat = numpy.zeros(9).reshape(3, 3)
        mujoco.mju_quat2Mat(rot_mat.ravel(), self._body.get_xquat())
        return rot_mat

    def get_direction(self) -> numpy.ndarray:
        a = numpy.dot(self.get_orientation(), [0.0, 1.0, 0.0])
        d = numpy.linalg.norm(a, ord=2)
        return a / d

    def rotate_wheel(self, left, right):
        self._left_act.ctrl = 1000 * left
        self._right_act.ctrl = 1000 * right

    def act(
            self,
            pheromone_value: float,
            nest_pos: numpy.ndarray,
            robot_pos: list[numpy.ndarray],
            feed_pos: list[numpy.ndarray]
    ):
        from environments import sensor

        pos = self.get_pos()
        mat = self.get_orientation()

        ref_nest_pos = numpy.dot(numpy.linalg.inv(mat), nest_pos - pos)[:2]

        sensed_robots_position = sensor.OmniSensor(pos, mat, 17.5, 70)
        for rp in robot_pos:
            sensed_robots_position.sense(rp)

        sensed_feeds_position = sensor.OmniSensor(pos, mat, 17.5 + 50.0, 70)
        for fp in feed_pos:
            sensed_feeds_position.sense(fp)

        input_ = numpy.concatenate(
            [ref_nest_pos, sensed_robots_position.value, sensed_feeds_position.value, [pheromone_value]]
        )
        ctrl = self.brain.calc(input_)

        # import colorsys
        # action_index: int = numpy.argmax(self.brain.get_calced_feature_value())
        # self._state_ball.set_pos([pos[0], pos[1], 5])
        #
        # pattern = {
        #     31: 0, 41: 1, 13: 2, 29: 3, 17: 4,
        #     14: 5, 43: 6, 33: 7, 44: 8, 9: 9,
        #     16: 10, 27: 11, 42: 12, 15: 13, 39: 14
        # }
        # if action_index not in pattern.keys():
        #     action_index = self._prev_action
        #     ctrl = numpy.tanh(self.brain.get_action(action_index))
        #
        # self._prev_action = action_index
        # tmp = colorsys.hsv_to_rgb(0.66 * (1.0 - pattern[action_index] / len(pattern)), 1.0, 1.0)
        # self._state_ball.set_rgba([tmp[0], tmp[1], tmp[2], 1.0])

        self.rotate_wheel(ctrl[0], ctrl[1])
        return 30 * ctrl[2]


class Environment(optimizer.MuJoCoEnvInterface):
    def __init__(
            self,
            brain: RobotBrain,
            nest_pos: (float, float),
            robot_pos: list[(float, float)],
            feed_pos: list[(float, float)],
            sv: float,
            evaporate: float,
            diffusion: float,
            decrease: float,
            pheromone_field_panel_size: float,
            pheromone_field_pos: (float, float),
            pheromone_field_shape: (int, int),
            timestep: int,
            window: miscellaneous.Window = None,
            camera: wrap_mjc.Camera = None
    ):
        from environments import pheromone

        self.timestep = timestep
        self.window = window
        self.camera = camera

        robot_pos = [(rp[0], rp[1], 45.0 if i < len(robot_pos) / 2 else -45.0) for i, rp in enumerate(robot_pos)]

        xml = _gen_env(
            nest_pos, robot_pos, feed_pos,
            pheromone_field_panel_size, pheromone_field_pos, pheromone_field_shape
        )
        self.model = wrap_mjc.WrappedModel(xml)

        if camera is not None:
            self.model.set_camera(camera)

        self.pheromone_field = pheromone.PheromoneField(
            pheromone_field_pos[0], pheromone_field_pos[1],
            pheromone_field_panel_size, 1,
            pheromone_field_shape[0], pheromone_field_shape[1],
            sv, evaporate, diffusion, decrease,
            None if window is None else self.model
        )

        self.robots = [_Robot(copy.deepcopy(brain), self.model, i) for i in range(0, len(robot_pos))]
        self.feeds = [_Feed(self.model, i) for i in range(0, len(feed_pos))]
        self.nest_pos = numpy.array([nest_pos[0], nest_pos[1], 0])

        self.loss = 0.0

    def calc_step(self) -> float:
        # Calculate
        self.model.step()
        for _ in range(5):
            self.pheromone_field.update_cells(0.033333 / 5)

        # Stop unstable state
        z_axis = numpy.array([0, 0, 1])
        for r in self.robots:
            c = numpy.dot(z_axis, r.get_direction())
            if not (-0.5 < c < 0.5):
                return float("inf")

        # Act
        robot_pos = [r.get_pos() for r in self.robots]
        feed_pos = [f.get_pos() for f in self.feeds]
        for r, rp in zip(self.robots, robot_pos):
            pheromone_value = self.pheromone_field.get_gas(rp[0], rp[1])
            secretion = r.act(pheromone_value, self.nest_pos, robot_pos, feed_pos)
            self.pheromone_field.add_liquid(rp[0], rp[1], secretion)

        # Calculate loss
        feed_range = 50000.0
        dt_loss_feed_nest = 0.0
        dt_loss_feed_robot = 0.0
        for f, fp in zip(self.feeds, feed_pos):
            feed_nest_distance = numpy.linalg.norm((self.nest_pos - fp)[0:2], ord=2)
            init_dist = numpy.linalg.norm((f.get_init_pos() - self.nest_pos)[0:2], ord=2)
            dt_loss_feed_nest += feed_nest_distance / init_dist

            for rp in robot_pos:
                d = numpy.sum((fp[0:2] - rp[0:2]) ** 2)
                dt_loss_feed_robot -= numpy.exp(-d / feed_range)

        dt_loss_feed_nest *= 1.0 / len(self.feeds)
        dt_loss_feed_robot *= 1e-7 / (len(self.feeds) * len(self.robots))
        self.loss += dt_loss_feed_nest + dt_loss_feed_robot

        return self.loss

    def calc(self) -> float:
        for _ in range(0, 5):
            self.model.step()

        for t in range(0, self.timestep):
            score = self.calc_step()
            if numpy.isinf(score):
                return score

        return self.loss

    def render(self):
        if self.window is not None:
            if not self.window.render(self.model, self.camera):
                exit()
            self.pheromone_field.update_panels()
            self.model.draw_text(f"{self.loss}", 0, 0, (1, 1, 1))
            self.window.flush()

    def calc_and_show(self) -> float:
        for _ in range(0, 5):
            self.model.step()

        for t in range(0, self.timestep):
            score = self.calc_step()
            if numpy.isinf(score):
                return score
            self.render()

        return self.loss


class EnvCreator(optimizer.MuJoCoEnvCreator):
    def __init__(self):
        self.nest_pos: (float, float) = (0, 0)
        self.robot_pos: list[(float, float)] = [(0, 0)]
        self.feed_pos: list[(float, float)] = [(0, 0)]
        self.sv: float = 0.0
        self.evaporate: float = 0.0
        self.diffusion: float = 0.0
        self.decrease: float = 0.0
        self.pheromone_field_panel_size: float = 0.0
        self.pheromone_field_pos: (float, float) = (0, 0)
        self.pheromone_field_shape: (int, int) = (0, 0)
        self.timestep: int = 100

    def save(self):
        import struct

        packed = [struct.pack("<dd", self.nest_pos[0], self.nest_pos[1])]

        packed.extend([struct.pack("<I", len(self.robot_pos))])
        packed.extend([struct.pack("<dd", p[0], p[1]) for p in self.robot_pos])

        packed.extend([struct.pack("<I", len(self.feed_pos))])
        packed.extend([struct.pack("<dd", p[0], p[1]) for p in self.feed_pos])

        packed.extend([struct.pack("<dddd", self.sv, self.evaporate, self.diffusion, self.decrease)])

        packed.extend([struct.pack("<d", self.pheromone_field_panel_size)])
        packed.extend([struct.pack("<dd", self.pheromone_field_pos[0], self.pheromone_field_pos[1])])
        packed.extend([struct.pack("<II", self.pheromone_field_shape[0], self.pheromone_field_shape[1])])

        packed.extend([struct.pack("<I", self.timestep)])

        return b"".join(packed)

    def load(self, data: bytes, offset: int = 0) -> int:
        import struct

        # 巣の座標
        s = offset
        e = 16
        self.nest_pos = struct.unpack("<dd", data[s:e])[0:2]

        # ロボットの座標
        s = e
        e = s + 4
        num = struct.unpack("<I", data[s:e])[0]
        self.robot_pos.clear()
        for _ in range(0, num):
            s = e
            e = s + 16
            rp = struct.unpack("<dd", data[s:e])[0:2]
            self.robot_pos.append(rp)

        # 餌の座標
        s = e
        e = s + 4
        num = struct.unpack("<I", data[s:e])[0]
        self.feed_pos.clear()
        for _ in range(0, num):
            s = e
            e = s + 16
            fp = struct.unpack("<dd", data[s:e])[0:2]
            self.feed_pos.append(fp)

        # フェロモンの設定
        s = e
        e = s + 8 + 8 + 8 + 8
        self.sv, self.evaporate, self.diffusion, self.decrease = struct.unpack("<dddd", data[s:e])[0:4]

        # MuJoCo上でのフェロモンのパネルの大きさ
        s = e
        e = s + 8
        self.pheromone_field_panel_size = struct.unpack("<d", data[s:e])[0]

        # MuJoCo上でのフェロモンのパネルの座標
        s = e
        e = s + 16
        self.pheromone_field_pos = struct.unpack("<dd", data[s:e])[0:2]

        # MuJoCo上でのフェロモンパネルの数
        s = e
        e = s + 8
        self.pheromone_field_shape = struct.unpack("<II", data[s:e])[0:2]

        # MuJoCoのタイムステップ
        s = e
        e = s + 4
        self.timestep = struct.unpack("<I", data[s:e])[0]

        return e - offset

    def dim(self) -> int:
        return RobotBrain(None).num_dim()

    def create(self, para) -> Environment:
        brain = RobotBrain(para)
        return Environment(
            brain,
            self.nest_pos,
            self.robot_pos,
            self.feed_pos,
            self.sv,
            self.evaporate,
            self.diffusion,
            self.decrease,
            self.pheromone_field_panel_size,
            self.pheromone_field_pos,
            self.pheromone_field_shape,
            self.timestep,
            None,
            None
        )

    def create_mujoco_env(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> Environment:
        brain = RobotBrain(para)
        return Environment(
            brain,
            self.nest_pos,
            self.robot_pos,
            self.feed_pos,
            self.sv,
            self.evaporate,
            self.diffusion,
            self.decrease,
            self.pheromone_field_panel_size,
            self.pheromone_field_pos,
            self.pheromone_field_shape,
            self.timestep,
            window,
            camera
        )
