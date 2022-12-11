import copy
from typing import Sequence

import mujoco
import numpy

from studyLib import optimizer, wrap_mjc, miscellaneous, nn_tools


def _gen_env(
        nest_pos: (float, float),
        robot_pos: list[(float, float, float)],
        obstacle_pos: list[(float, float)],
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
    # Create Setting for Obstacles
    ######################################################################################################
    obstacle_default = default.add_default("obstacles")
    obstacle_default.add_geom({
        "type": "cylinder",
        "size": "60 10",
        "rgba": "1 0 0 1",
        "condim": "1",
        "priority": "1",
        "friction": "0.5 0.0 0.0",
        "contype": "1",
        "conaffinity": "2",
    })

    ######################################################################################################
    # Create Setting for Feeds
    ######################################################################################################
    feed_default = default.add_default("feeds")
    feed_default.add_geom({
        "type": "cylinder",
        "size": "35 10",
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
    # Create Obstacles
    ######################################################################################################
    for i, op in enumerate(obstacle_pos):
        worldbody.add_geom({
            "name": f"obstacle{i}",
            "class": "obstacles",
            "pos": f"{op[0]} {op[1]} 10",
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


class _Obstacle:
    def __init__(self, model: wrap_mjc.WrappedModel, number: int):
        self._pos = model.get_geom(f"obstacle{number}").get_pos()

    def get_pos(self):
        return self._pos.copy()


class _Feed:
    def __init__(self, model: wrap_mjc.WrappedModel, number: int):
        self._body = model.get_body(f"feed{number}")
        self._velocity_sensor = model.get_sensor(f"sensor_feed{number}_velocity")

    def get_pos(self):
        return self._body.get_xpos().copy()

    def get_velocity(self) -> numpy.ndarray:
        return self._velocity_sensor.get_data()


class ConvertPheromone(nn_tools.interface.CalcActivator):
    def __init__(self, num_node: int, pheromone_indexes: len[int]):
        super().__init__(num_node)
        self.mask = [False] * num_node
        for i in pheromone_indexes:
            self.mask[i] = True

    def calc(self, input_: nn_tools.la.ndarray, output: nn_tools.la.ndarray) -> int:
        nn_tools.la.copyto(output, input_)
        output[self.mask] += 1.0
        output[self.mask] *= 0.5
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        return None


class RobotBrain:
    def __init__(self, para):
        self._calculator = nn_tools.Calculator(10)
        self._calculator.add_layer(nn_tools.BufLayer(10))  # 0

        pheromone_calculator = nn_tools.Calculator(10)  # 1->0
        pheromone_calculator.add_layer(nn_tools.FilterLayer([8, 9]))  # 1->0->0
        pheromone_calculator.add_layer(nn_tools.AffineLayer(30))  # 1->0->1
        pheromone_calculator.add_layer(nn_tools.TanhLayer(30))  # 1->0->2
        pheromone_calculator.add_layer(nn_tools.AffineLayer(10))  # 1->0->3
        pheromone_calculator.add_layer(nn_tools.TanhLayer(10))  # 1->0->4
        pheromone_calculator.add_layer(nn_tools.AffineLayer(4))  # 1->0->5

        state_calculator = nn_tools.Calculator(10)  # 1->1
        state_calculator.add_layer(nn_tools.FilterLayer([0, 1, 2, 3, 4, 5, 6, 7]))  # 1->1->0
        state_calculator.add_layer(nn_tools.AffineLayer(10))  # 1->1->1
        state_calculator.add_layer(nn_tools.TanhLayer(10))  # 1->1->2
        state_calculator.add_layer(nn_tools.AffineLayer(4))  # 1->1->3

        self._calculator.add_layer(nn_tools.ParallelLayer([  # 1
            pheromone_calculator,  # 1->0
            state_calculator  # 1->1
        ]))
        self._calculator.add_layer(nn_tools.BufLayer(8))  # 2
        self._calculator.add_layer(nn_tools.AddFoldLayer(4))  # 3
        self._calculator.add_layer(nn_tools.TanhLayer(4))  # 4
        self._calculator.add_layer(ConvertPheromone(4, [2, 3]))  # 5
        self._calculator.add_layer(nn_tools.BufLayer(4))  # 6

        if not (para is None):
            self._calculator.load(para)

    def num_dim(self):
        return self._calculator.num_dim()

    def calc(self, array):
        return self._calculator.calc(array)

    def get_input(self) -> numpy.ndarray:
        buf_layer: nn_tools.BufLayer = self._calculator.get_layer(0)
        return buf_layer.buf.copy()

    def get_feature_values(self) -> numpy.ndarray:
        buf_layer: nn_tools.BufLayer = self._calculator.get_layer(2)
        return buf_layer.buf.copy()

    def get_output(self) -> numpy.ndarray:
        buf_layer: nn_tools.BufLayer = self._calculator.get_layer(6)
        return buf_layer.buf.copy()

    def get_pheromone_calculator(self) -> nn_tools.Calculator:
        parallel_layer: nn_tools.ParallelLayer = self._calculator.get_layer(1)
        return parallel_layer.calcs[0]


class _Robot:
    def __init__(self, brain: RobotBrain, model: wrap_mjc.WrappedModel, number: int):
        self.brain = brain
        self._body = model.get_body(f"robot{number}")
        self._left_act = model.get_act(f"act_robot{number}_left")
        self._right_act = model.get_act(f"act_robot{number}_right")

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
            pheromone_values: len[float],
            nest_pos: numpy.ndarray,
            robot_pos: list[numpy.ndarray],
            obstacle_pos: list[numpy.ndarray],
            feed_pos: list[numpy.ndarray]
    ):
        from environments import sensor

        pos = self.get_pos()
        mat = self.get_orientation()

        ref_nest_pos = numpy.dot(numpy.linalg.inv(mat), nest_pos - pos)[:2]

        rs = sensor.OmniSensor(pos, mat, 17.5, 70)
        for rp in robot_pos:
            rs.sense(rp)

        os = sensor.OmniSensor(pos, mat, 17.5 + 60.0, 70)
        for op in obstacle_pos:
            os.sense(op)

        fs = sensor.OmniSensor(pos, mat, 17.5 + 50.0, 70)
        for fp in feed_pos:
            fs.sense(fp)

        input_ = numpy.concatenate(
            [ref_nest_pos, rs.value, os.value, fs.value, pheromone_values]
        )
        ctrl = self.brain.calc(input_)

        self.rotate_wheel(ctrl[0], ctrl[1])
        return 30 * ctrl[2], 30 * ctrl[3]


class Environment(optimizer.MuJoCoEnvInterface):
    def __init__(
            self,
            brain: RobotBrain,
            nest_pos: (float, float),
            robot_pos: list[(float, float)],
            obstacle_pos: list[(float, float)],
            feed_pos: list[(float, float)],
            sv: len[float],
            evaporate: len[float],
            diffusion: len[float],
            decrease: len[float],
            pheromone_field_panel_size: float,
            pheromone_field_pos: (float, float),
            pheromone_field_shape: (int, int),
            timestep: int,
            show_pheromone_index: int = 0,
            window: miscellaneous.Window = None,
            camera: wrap_mjc.Camera = None
    ):
        from environments import pheromone

        self.timestep = timestep
        self.show_pheromone_index = show_pheromone_index
        self.window = window
        self.camera = camera

        robot_pos = [(rp[0], rp[1], 0) for i, rp in enumerate(robot_pos)]

        xml = _gen_env(
            nest_pos, robot_pos, obstacle_pos, feed_pos,
            pheromone_field_panel_size, pheromone_field_pos, pheromone_field_shape
        )
        self.model = wrap_mjc.WrappedModel(xml)

        if camera is not None:
            self.model.set_camera(camera)

        self.pheromone_field = []
        if len(sv) != len(evaporate) != len(diffusion) != len(decrease):
            raise "Invalid pheromone parameter."
        for i in range(0, len(sv)):
            self.pheromone_field.append(pheromone.PheromoneField(
                pheromone_field_pos[0], pheromone_field_pos[1],
                pheromone_field_panel_size, 1,
                pheromone_field_shape[0], pheromone_field_shape[1],
                sv[i], evaporate[i], diffusion[i], decrease[i],
                self.model if (window is not None and i == self.show_pheromone_index) else None
            ))

        self.robots = [_Robot(copy.deepcopy(brain), self.model, i) for i in range(0, len(robot_pos))]
        self.obstacles = [_Obstacle(self.model, i) for i in range(0, len(obstacle_pos))]
        self.feeds = [_Feed(self.model, i) for i in range(0, len(feed_pos))]
        self.nest_pos = numpy.array([nest_pos[0], nest_pos[1], 0])
        self.obstacle_pos = [o.get_pos() for o in self.obstacles]

        self.loss = 0.0

        for _ in range(0, 5):
            self.model.step()

    def calc_step(self) -> float:
        # Calculate
        self.model.step()
        for pf in self.pheromone_field:
            for _ in range(5):
                pf.update_cells(0.033333 / 5)

        # Stop unstable state
        z_axis = numpy.array([0, 0, 1])
        for r in self.robots:
            c = numpy.dot(z_axis, r.get_direction())
            if not (-0.5 < c < 0.5):
                print("Catch warning!")
                return float("inf")
            if self.model.count_raised_warning() > 0:
                print("Catch warning!")
                return float("inf")

        # Act
        robot_pos = [r.get_pos() for r in self.robots]
        feed_pos = [f.get_pos() for f in self.feeds]
        for r, rp in zip(self.robots, robot_pos):
            pheromone_values = [pf.get_gas(rp[0], rp[1]) for pf in self.pheromone_field]
            secretion = r.act(pheromone_values, self.nest_pos, robot_pos, self.obstacle_pos, feed_pos)

            for i, s in enumerate(secretion):
                self.pheromone_field[i].add_liquid(rp[0], rp[1], s)

        # Calculate loss
        feed_range = 10000.0
        dt_loss_feed_nest = 0.0
        dt_loss_feed_robot = 0.0
        for f, fp in zip(self.feeds, feed_pos):
            feed_nest_vector = (self.nest_pos - fp)[0:2]
            feed_nest_distance = numpy.linalg.norm(feed_nest_vector, ord=2)
            valid_feed_velocity = numpy.dot(feed_nest_vector / feed_nest_distance, f.get_velocity()[0:2])
            dt_loss_feed_nest -= valid_feed_velocity

            for rp in robot_pos:
                d = numpy.sum((fp[0:2] - rp[0:2]) ** 2)
                dt_loss_feed_robot -= numpy.exp(-d / feed_range)

        obstacle_range = 200.0
        dt_loss_obstacle_robot = 0.0
        for rp in robot_pos:
            for op in self.obstacle_pos:
                d = numpy.sum((rp[0:2] - op[0:2]) ** 2)
                dt_loss_obstacle_robot += numpy.exp(-d / obstacle_range)

        dt_loss_feed_nest *= 0.1 / len(self.feeds)
        dt_loss_feed_robot *= 1.0 / (len(self.feeds) * len(self.robots))
        dt_loss_obstacle_robot *= 1e12 / (len(self.obstacle_pos) * len(self.robots))
        self.loss += dt_loss_feed_nest + dt_loss_feed_robot + dt_loss_obstacle_robot

        return self.loss

    def calc(self) -> float:
        for t in range(0, self.timestep):
            score = self.calc_step()
            if numpy.isinf(score):
                return score
        return self.loss

    def render(self):
        if self.window is not None:
            if not self.window.render(self.model, self.camera):
                exit()
            self.pheromone_field[self.show_pheromone_index].update_panels()
            self.model.draw_text(f"{self.loss}", 0, 0, (1, 1, 1))
            self.window.flush()

    def calc_and_show(self) -> float:
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
        self.obstacle_pos: list[(float, float)] = [(0, 0)]
        self.feed_pos: list[(float, float)] = [(0, 0)]
        self.sv: list[float] = [0.0]
        self.evaporate: list[float] = [0.0]
        self.diffusion: list[float] = [0.0]
        self.decrease: list[float] = [0.0]
        self.pheromone_field_panel_size: float = 0.0
        self.pheromone_field_pos: (float, float) = (0, 0)
        self.pheromone_field_shape: (int, int) = (0, 0)
        self.show_pheromone_index: int = 0
        self.timestep: int = 100

    def save(self):
        import struct

        packed = [struct.pack("<dd", self.nest_pos[0], self.nest_pos[1])]

        packed.extend([struct.pack("<I", len(self.robot_pos))])
        packed.extend([struct.pack("<dd", p[0], p[1]) for p in self.robot_pos])

        packed.extend([struct.pack("<I", len(self.obstacle_pos))])
        packed.extend([struct.pack("<dd", p[0], p[1]) for p in self.obstacle_pos])

        packed.extend([struct.pack("<I", len(self.feed_pos))])
        packed.extend([struct.pack("<dd", p[0], p[1]) for p in self.feed_pos])

        packed.extend([struct.pack("<I", len(self.sv))])
        for i in range(0, len(self.sv)):
            packed.extend([struct.pack("<dddd", self.sv[i], self.evaporate[i], self.diffusion[i], self.decrease[i])])

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

        # 障害物の座標
        s = e
        e = s + 4
        num = struct.unpack("<I", data[s:e])[0]
        self.obstacle_pos.clear()
        for _ in range(0, num):
            s = e
            e = s + 16
            op = struct.unpack("<dd", data[s:e])[0:2]
            self.obstacle_pos.append(op)

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
        e = s + 4
        num = struct.unpack("<I", data[s:e])[0]
        self.sv.clear()
        self.evaporate.clear()
        self.diffusion.clear()
        self.decrease.clear()
        for _ in range(0, num):
            s = e
            e = s + 32
            sv, evaporate, diffusion, decrease = struct.unpack("<dddd", data[s:e])[0:4]
            self.sv.append(sv)
            self.evaporate.append(evaporate)
            self.diffusion.append(diffusion)
            self.decrease.append(decrease)

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
            self.obstacle_pos,
            self.feed_pos,
            self.sv,
            self.evaporate,
            self.diffusion,
            self.decrease,
            self.pheromone_field_panel_size,
            self.pheromone_field_pos,
            self.pheromone_field_shape,
            self.timestep,
            self.show_pheromone_index,
            None,
            None
        )

    def create_mujoco_env(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> Environment:
        brain = RobotBrain(para)
        return Environment(
            brain,
            self.nest_pos,
            self.robot_pos,
            self.obstacle_pos,
            self.feed_pos,
            self.sv,
            self.evaporate,
            self.diffusion,
            self.decrease,
            self.pheromone_field_panel_size,
            self.pheromone_field_pos,
            self.pheromone_field_shape,
            self.timestep,
            self.show_pheromone_index,
            window,
            camera
        )
