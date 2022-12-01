import mujoco
import numpy

from studyLib import optimizer, wrap_mjc, miscellaneous, nn_tools
from studyLib.optimizer import EnvInterface, MuJoCoEnvInterface


def _gen_env(
        nest_pos: (float, float),
        robot_pos: list[(float, float)],
        obstacle_pos: list[(float, float)],
        feed_pos: list[(float, float)],
        pheromone_field_panel_size: float,
        pheromone_field_pos: (float, float),
        pheromone_field_shape: (int, int)
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
    sensor = generator.add_sensor()

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

    # Create Wall
    pheromone_field_size = (
        pheromone_field_panel_size * pheromone_field_shape[0], pheromone_field_panel_size * pheromone_field_shape[1]
    )
    worldbody.add_geom({
        "type": "box",
        "pos": f"{pheromone_field_pos[0] + 0.5 * (pheromone_field_size[0] + pheromone_field_panel_size)} {pheromone_field_pos[1]} 5",
        "size": f"{pheromone_field_panel_size * 0.5} {pheromone_field_size[1] * 0.5} 5",
        "rgba": "0.7 0.7 0.7 0.5",
    })
    worldbody.add_geom({
        "type": "box",
        "pos": f"{pheromone_field_pos[0] - 0.5 * (pheromone_field_size[0] + pheromone_field_panel_size)} {pheromone_field_pos[1]} 5",
        "size": f"{pheromone_field_panel_size * 0.5} {pheromone_field_size[1] * 0.5} 5",
        "rgba": "0.7 0.7 0.7 0.5",
    })
    worldbody.add_geom({
        "type": "box",
        "pos": f"{pheromone_field_pos[0]} {pheromone_field_pos[1] + 0.5 * (pheromone_field_size[1] + pheromone_field_panel_size)} 5",
        "size": f"{pheromone_field_size[0] * 0.5} {pheromone_field_panel_size * 0.5} 5",
        "rgba": "0.7 0.7 0.7 0.5",
    })
    worldbody.add_geom({
        "type": "box",
        "pos": f"{pheromone_field_pos[0]} {pheromone_field_pos[1] - 0.5 * (pheromone_field_size[1] + pheromone_field_panel_size)} 5",
        "size": f"{pheromone_field_size[0] * 0.5} {pheromone_field_panel_size * 0.5} 5",
        "rgba": "0.7 0.7 0.7 0.5",
    })

    # Create Nest
    worldbody.add_geom({
        "type": "cylinder",
        "pos": f"{nest_pos[0]} {nest_pos[1]} -3",
        "size": "50 1",
        "rgba": "0.0 1.0 0.0 1",
    })

    # Create Obstacles
    for i, op in enumerate(obstacle_pos):
        worldbody.add_geom({
            "name": f"obstacle{i}",
            "type": "cylinder",
            "size": "50 10",
            "rgba": "1 0 0 1",
            "pos": f"{op[0]} {op[1]} 10",
            "contype": "1",
            "conaffinity": "2"
        })

    # Create Feeds
    feed_weight = 25000
    for i, fp in enumerate(feed_pos):
        feed_body = worldbody.add_body({
            "name": f"feed{i}",
            "pos": f"{fp[0]} {fp[1]} 11"
        })
        feed_body.add_freejoint()
        feed_body.add_site({"name": f"site_feed{i}"})
        feed_body.add_geom({
            "type": "cylinder",
            "size": "100 10",
            "mass": f"{feed_weight}",
            "rgba": "0 1 1 1",
            "condim": "3",
            "contype": "1",
            "conaffinity": "1",
            "priority": "1",
            "friction": "0.5 0.0 0.0"
        })
        sensor.add_velocimeter({
            "name": f"sensor_feed{i}_velocity",
            "site": f"site_feed{i}"
        })

    # Create Robots
    depth = 1.0
    body_density = 0.51995  # 鉄の密度(7.874 g/cm^3), ルンバの密度(0.51995 g/cm^3)
    wheel_density = 0.3
    for i, rp in enumerate(robot_pos):
        robot_body = worldbody.add_body({
            "name": f"robot{i}",
            "pos": f"{rp[0]} {rp[1]} {10 + depth + 0.5}",
            "axisangle": f"0 0 1 0",
        })
        robot_body.add_freejoint()
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
        right_wheel_body.add_joint({"name": f"joint_robot{i}_right", "type": "hinge", "axis": "-1 0 0"})
        right_wheel_body.add_geom({
            "type": "cylinder",
            "size": "5 5",
            "density": f"{wheel_density}",
            "axisangle": "0 1 0 90",
            "condim": "6",
            "contype": "1",
            "conaffinity": "1",
            "priority": "1",
        })

        left_wheel_body = robot_body.add_body({"pos": f"-10 0 -{depth}"})
        left_wheel_body.add_joint({"name": f"joint_robot{i}_left", "type": "hinge", "axis": "-1 0 0"})
        left_wheel_body.add_geom({
            "type": "cylinder",
            "size": "5 5",
            "density": f"{wheel_density}",
            "axisangle": "0 1 0 90",
            "condim": "6",
            "contype": "1",
            "conaffinity": "1",
            "priority": "1",
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
            "name": f"act_robot{i}_left",
            "joint": f"joint_robot{i}_left",
            "gear": "80",
            "ctrllimited": "true",
            "ctrlrange": "-50000 50000",
            "kv": "1"
        })
        act.add_velocity({
            "name": f"act_robot{i}_right",
            "joint": f"joint_robot{i}_right",
            "gear": "80",
            "ctrllimited": "true",
            "ctrlrange": "-50000 50000",
            "kv": "1"
        })

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


class RobotBrain:
    def __init__(self, para):
        self._calculator = nn_tools.Calculator(11)

        self._calculator.add_layer(nn_tools.AffineLayer(50))
        self._calculator.add_layer(nn_tools.TanhLayer(50))

        self._calculator.add_layer(nn_tools.AffineLayer(20))
        self._calculator.add_layer(nn_tools.TanhLayer(20))

        self._calculator.add_layer(nn_tools.AffineLayer(50))
        self._calculator.add_layer(nn_tools.IsMaxLayer(50))

        self._calculator.add_layer(nn_tools.InnerDotLayer(3))
        self._calculator.add_layer(nn_tools.TanhLayer(3))

        if not (para is None):
            self._calculator.load(para)

    def num_dim(self):
        return self._calculator.num_dim()

    def calc(self, array):
        return self._calculator.calc(array)

    def num_pattern(self) -> int:
        rbf: nn_tools.GaussianRadialBasisLayer = self._calculator.get_layer(0)
        return rbf.weights.shape[0]

    def get_pattern(self, i: int) -> (numpy.ndarray, float):
        rbf: nn_tools.GaussianRadialBasisLayer = self._calculator.get_layer(0)
        return numpy.array(rbf.centroid[i]), numpy.abs(rbf.weights[i])

    def get_action(self, i: int) -> numpy.ndarray:
        inner_dot_layer: nn_tools.InnerDotLayer = self._calculator.get_layer(2)
        return numpy.array(inner_dot_layer.weights[:, i])


class _Robot:
    def __init__(self, brain: RobotBrain, model: wrap_mjc.WrappedModel, number: int):
        self._brain = brain
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
        a = numpy.dot(self.get_orientation(), [0.0, 1.0, 0.0])[0:2]
        d = numpy.linalg.norm(a, ord=2)
        return a / d

    def act(
            self,
            pheromone_value: float,
            pheromone_grad: numpy.ndarray,
            nest_pos: numpy.ndarray,
            robot_pos: list[numpy.ndarray],
            obstacle_pos: list[numpy.ndarray],
            feed_pos: list[numpy.ndarray]
    ):
        from environments import sensor

        pos = self.get_pos()
        mat = self.get_orientation()

        ref_nest_pos = (nest_pos - pos)[:2]
        nest_dist = numpy.linalg.norm(ref_nest_pos, ord=2)
        if nest_dist > 0.0:
            nest_direction = ref_nest_pos / nest_dist
        else:
            nest_direction = numpy.zeros(2)

        rs = sensor.OmniSensor(pos, mat, 100)
        for rp in robot_pos:
            rs.sense(rp)

        os = sensor.OmniSensor(pos, mat, 150)
        for op in obstacle_pos:
            os.sense(op)

        fs = sensor.OmniSensor(pos, mat, 200)
        for fp in feed_pos:
            fs.sense(fp)

        input_ = numpy.concatenate(
            [nest_direction, rs.value, os.value, fs.value, [pheromone_value], pheromone_grad]
        )
        ctrl = self._brain.calc(input_)

        self._left_act.ctrl = 10000 * ctrl[0]
        self._right_act.ctrl = 10000 * ctrl[1]
        return 30 * (ctrl[2] + 1.0) * 0.5


def _evaluate(
        brain: RobotBrain,
        nest_pos: (float, float),
        robot_pos: list[(float, float)],
        obstacle_pos: list[(float, float)],
        feed_pos: list[(float, float)],
        sv: float,
        evaporate: float,
        diffusion: float,
        decrease: float,
        pheromone_field_panel_size: float,
        pheromone_field_pos: (float, float),
        pheromone_field_shape: (int, int),
        timestep: int,
        camera: wrap_mjc.Camera = None,
        window: miscellaneous.Window = None
) -> float:
    from environments import pheromone

    xml = _gen_env(
        nest_pos, robot_pos, obstacle_pos, feed_pos,
        pheromone_field_panel_size, pheromone_field_pos, pheromone_field_shape
    )
    model = wrap_mjc.WrappedModel(xml)

    if camera is not None:
        model.set_camera(camera)

    pheromone_field = pheromone.PheromoneField(
        pheromone_field_pos[0], pheromone_field_pos[1],
        pheromone_field_panel_size, 1,
        pheromone_field_shape[0], pheromone_field_shape[1],
        sv, evaporate, diffusion, decrease,
        None if window is None else model
    )

    robots = [_Robot(brain, model, i) for i in range(0, len(robot_pos))]
    obstacles = [_Obstacle(model, i) for i in range(0, len(obstacle_pos))]
    feeds = [_Feed(model, i) for i in range(0, len(feed_pos))]
    nest_pos = numpy.array([nest_pos[0], nest_pos[1], 0])
    obstacle_pos = [o.get_pos() for o in obstacles]

    for _ in range(0, 5):
        model.step()

    loss = 0.0
    for t in range(0, timestep):
        # Simulate
        robot_pos = [r.get_pos() for r in robots]
        feed_pos = [f.get_pos() for f in feeds]
        for r, rp in zip(robots, robot_pos):
            pheromone_value = pheromone_field.get_gas(rp[0], rp[1])
            pheromone_grad = pheromone_field.get_gas_grad(rp, r.get_direction())
            secretion = r.act(pheromone_value, pheromone_grad, nest_pos, robot_pos, obstacle_pos, feed_pos)
            pheromone_field.add_liquid(rp[0], rp[1], secretion)

        model.step()
        for _ in range(5):
            pheromone_field.update_cells(0.033333 / 5)

        # Render MuJoCo Scene
        if window is not None:
            if not window.render(model, camera):
                exit()
            pheromone_field.update_panels()
            model.draw_text(f"{loss}", 0, 0, (0, 0, 0))
            window.flush()

        # Calculate loss
        feed_range_bias = 300000.0
        feed_range_esp = 20000.0
        feed_robot_loss = 0.0
        feed_nest_loss = 0.0
        for f in feeds:
            fv = numpy.linalg.norm(f.get_velocity()[0:2], ord=2)
            fp = f.get_pos()

            for r in robots:
                d = numpy.sum((fp[0:2] - r.get_pos()[0:2]) ** 2)
                feed_robot_loss -= numpy.exp(-d / (feed_range_bias * fv + feed_range_esp))

            feed_nest_loss += numpy.linalg.norm(fp[0:2] - nest_pos[0:2], ord=2)

        feed_robot_loss /= (len(feeds) * len(robots))
        feed_nest_loss *= 0.001 / len(feeds)
        loss += feed_robot_loss + feed_nest_loss

    return loss


class Environment(optimizer.MuJoCoEnvInterface):
    def __init__(
            self,
            nest_pos: (float, float),
            robot_pos: list[(float, float)],
            obstacle_pos: list[(float, float)],
            feed_pos: list[(float, float)],
            sv: float,
            evaporate: float,
            diffusion: float,
            decrease: float,
            pheromone_field_panel_size: float,
            pheromone_field_pos: (float, float),
            pheromone_field_shape: (int, int),
            timestep: int
    ):
        self.nest_pos: (float, float) = nest_pos
        self.robot_pos: list[(float, float)] = robot_pos
        self.obstacle_pos: list[(float, float)] = obstacle_pos
        self.feed_pos: list[(float, float)] = feed_pos
        self.sv: float = sv
        self.evaporate: float = evaporate
        self.diffusion: float = diffusion
        self.decrease: float = decrease
        self.pheromone_field_panel_size: float = pheromone_field_panel_size
        self.pheromone_field_pos: (float, float) = pheromone_field_pos
        self.pheromone_field_shape: (int, int) = pheromone_field_shape
        self.timestep: int = timestep

    def calc(self, para) -> float:
        brain = RobotBrain(para)
        return _evaluate(
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
            None,
            None
        )

    def calc_and_show(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> float:
        brain = RobotBrain(para)
        return _evaluate(
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
            camera,
            window
        )


class EnvCreator(optimizer.MuJoCoEnvCreator):
    def __init__(self):
        self.nest_pos: (float, float) = (0, 0)
        self.robot_pos: list[(float, float)] = [(0, 0)]
        self.obstacle_pos: list[(float, float)] = [(0, 0)]
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

        packed.extend([struct.pack("<I", len(self.obstacle_pos))])
        packed.extend([struct.pack("<dd", p[0], p[1]) for p in self.obstacle_pos])

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

    def create(self) -> EnvInterface:
        return Environment(
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
            self.timestep
        )

    def create_mujoco_env(self) -> MuJoCoEnvInterface:
        return Environment(
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
            self.timestep
        )
