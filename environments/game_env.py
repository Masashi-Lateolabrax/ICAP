import copy

import mujoco
import numpy

from studyLib import optimizer, wrap_mjc, miscellaneous, nn_tools


def _gen_env(
        timestep: float,
        nest_pos: (float, float),
        robot_pos: list[(float, float, float)],
        obstacle_pos: list[(float, float)],
        feed_pos: list[(float, float)],
        pheromone_field_panel_size: float,
        pheromone_field_pos: (float, float),
        pheromone_field_shape: (int, int)
):
    generator = wrap_mjc.MuJoCoXMLGenerator("game_env")

    generator.add_option({
        "timestep": f"{timestep}",
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
    # Create Setting for Ground
    ######################################################################################################
    ground_default = default.add_default("ground")
    ground_default.add_geom({
        "type": "plane",
        "size": "0 0 0.05",
        "rgba": "1.0 1.0 1.0 0.0",
        "condim": "1",
        "priority": "0",
        "contype": "1",
        "conaffinity": "2",
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
    worldbody.add_geom({"class": "ground"})

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
        feed_body.add_site({"name": f"feed_center_site{i}"})
        feed_body.add_geom({"class": "feeds"})
        sensor.add_velocimeter({
            "name": f"feed{i}_velocity_sensor",
            "site": f"feed_center_site{i}"
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

        robot_camera = robot_body.add_camera({
            "name": f"robot{i}_camera",
            "pos": "0 17.5 0",
            "axisangle": "90 0 0"
        })

        right_wheel_body = robot_body.add_body({"pos": f"10 0 -{depth}"})
        right_wheel_body.add_joint({"name": f"robot{i}_right_joint", "type": "hinge", "axis": "-1 0 0"})
        right_wheel_body.add_geom({"class": "robot_wheel"})

        left_wheel_body = robot_body.add_body({"pos": f"-10 0 -{depth}"})
        left_wheel_body.add_joint({"name": f"robot{i}_left_joint", "type": "hinge", "axis": "-1 0 0"})
        left_wheel_body.add_geom({"class": "robot_wheel"})

        front_wheel_body = robot_body.add_body({"pos": f"0 15 {-5 + 1.5 - depth}"})
        front_wheel_body.add_joint({"type": "ball"})
        front_wheel_body.add_geom({"class": "robot_ball"})

        rear_wheel_body = robot_body.add_body({"pos": f"0 -15 {-5 + 1.5 - depth}"})
        rear_wheel_body.add_joint({"type": "ball"})
        rear_wheel_body.add_geom({"class": "robot_ball"})

        act.add_velocity({
            "name": f"robot{i}_left_act",
            "joint": f"robot{i}_left_joint",
            "kv": "100",
            "gear": "30",
        })
        act.add_velocity({
            "name": f"robot{i}_right_act",
            "joint": f"robot{i}_right_joint",
            "kv": "100",
            "gear": "30"
        })

    ######################################################################################################
    # Generate XML
    ######################################################################################################
    xml = generator.generate()
    return xml


class _Obstacle:
    def __init__(self, pos):
        self._pos = pos

    def get_pos(self):
        return self._pos.copy()


class _Feed:
    def __init__(self, body: wrap_mjc.WrappedBody, velocity_sensor):
        self._body = body
        self._velocity_sensor = velocity_sensor

    def get_pos(self):
        return self._body.get_xpos().copy()

    def get_velocity(self) -> numpy.ndarray:
        return self._velocity_sensor.get_data()


class RobotBrain:
    def __init__(self, para):
        self._calculator = nn_tools.Calculator(101)

        self._calculator.add_layer(nn_tools.ParallelLayer(
            [
                [
                    nn_tools.FilterLayer([i for i in range(0, 100)]),
                    nn_tools.AffineLayer(33),
                    nn_tools.TanhLayer(33),
                    nn_tools.AffineLayer(11),
                    nn_tools.TanhLayer(11),
                    nn_tools.AffineLayer(3),
                    nn_tools.IsMaxLayer(3)
                ],
                [
                    nn_tools.FilterLayer([100]),
                ]
            ]
        ))

        self._calculator.add_layer(nn_tools.AffineLayer(4))
        self._calculator.add_layer(nn_tools.TanhLayer(4))
        self._calculator.add_layer(nn_tools.AffineLayer(4))
        self._calculator.add_layer(nn_tools.TanhLayer(4))
        self._calculator.add_layer(nn_tools.AffineLayer(3))
        self._calculator.add_layer(nn_tools.SigmoidLayer(3))

        if para is not None:
            self._calculator.load(para)

    def num_dim(self):
        return self._calculator.num_dim()

    def calc(self, array):
        return self._calculator.calc(array)


class _Robot:
    def __init__(self, body: wrap_mjc.WrappedBody, left_act, right_act, sight):
        self._model = model
        self._body = body
        self._left_act = left_act
        self._right_act = right_act
        self._sight = sight

    def get_pos(self) -> numpy.ndarray:
        return self._body.get_xpos().copy()

    def get_orientation(self) -> numpy.ndarray:
        rot_mat = numpy.zeros(9).reshape(3, 3)
        mujoco.mju_quat2Mat(rot_mat.ravel(), self._body.get_xquat())
        return rot_mat

    def get_direction(self) -> numpy.ndarray:
        a = numpy.dot(self.get_orientation(), [0.0, 1.0, 0.0])
        a[2] = 0.0
        d = numpy.linalg.norm(a, ord=2)
        return a / d

    def get_sight(self):

    def rotate_wheel(self, left, right):
        self._left_act.ctrl = 1000 * left
        self._right_act.ctrl = 1000 * right

    def look(self):

    def act(
            self,
            vision: list[float],
            pheromone_values: list[float],
            nest_pos: numpy.ndarray,
    ):
        from environments import sensor

        wrap_mjc.

        pos = self.get_pos()
        mat = self.get_orientation()

        ref_nest_pos = numpy.dot(numpy.linalg.inv(mat), nest_pos - pos)[:2]

        rs = sensor.OmniSensor(pos, mat, 17.5, 70)
        for rp in robot_pos:
            rs.sense(rp)

        # os = sensor.OmniSensor(pos, mat, 17.5 + 60.0, 70)
        # for op in obstacle_pos:
        #     os.sense(op)

        fs = sensor.OmniSensor(pos, mat, 17.5 + 50.0, 70)
        for fp in feed_pos:
            fs.sense(fp)

        input_ = numpy.concatenate(
            [ref_nest_pos, rs.value, fs.value, pheromone_values]
        )
        ctrl = self.brain.calc(input_)

        self.rotate_wheel(ctrl[0], ctrl[1])
        return numpy.linalg.norm(ctrl, ord=2), (ctrl[2] * 30.0,)


class _World:
    def __init__(
            self,
            nest_pos: (float, float),
            robot_pos: list[(float, float)],
            obstacle_pos: list[(float, float)],
            feed_pos: list[(float, float)],
            sv: list[float],
            evaporate: list[float],
            diffusion: list[float],
            decrease: list[float],
            pheromone_field_panel_size: float,
            pheromone_field_pos: (float, float),
            pheromone_field_shape: (int, int),
            pheromone_iteration: int,
            timestep: float,
    ):
        self.timestep = timestep
        self.pheromone_iteration = pheromone_iteration
        self.num_robots = len(robot_pos)
        self.num_obstacles = len(obstacle_pos)
        self.num_feeds = len(feed_pos)

        xml = _gen_env(
            timestep,
            nest_pos, robot_pos, obstacle_pos, feed_pos,
            pheromone_field_panel_size, pheromone_field_pos, pheromone_field_shape
        )
        self.model = wrap_mjc.WrappedModel(xml)

        # Create Pheromone Field
        self.pheromone_field = []
        if len(sv) != len(evaporate) != len(diffusion) != len(decrease):
            raise "Invalid pheromone parameter."
        for i in range(0, len(sv)):
            self.pheromone_field.append(miscellaneous.pheromone.PheromoneField(
                pheromone_field_pos[0], pheromone_field_pos[1],
                pheromone_field_panel_size, 1,
                pheromone_field_shape[0], pheromone_field_shape[1],
                sv[i], evaporate[i], diffusion[i], decrease[i]
            ))

        # Create Pheromone Panels
        self.pheromone_panel = miscellaneous.PheromonePanels(
            self.model,
            pheromone_field_pos[0], pheromone_field_pos[1],
            pheromone_field_panel_size,
            pheromone_field_shape[0], pheromone_field_shape[1],
            0.05
        )

    def get_robot(self, index: int) -> _Robot:
        body = self.model.get_body(f"robot{index}")
        act_left = self.model.get_act(f"robot{index}_left_act")
        act_right = self.model.get_act(f"robot{index}_right_act")
        robot_camera = self.model.get_camera(f"robot{index}_camera")

        self.model.set_global_camera(
            robot_camera.get_xpos()
        )

        rot_mat = numpy.zeros(9).reshape(3, 3)
        mujoco.mju_quat2Mat(rot_mat.ravel(), body.get_xquat())
        inv_rot_mat = numpy.linalg.inv(rot_mat)

        convertor = lambda d, t, k1, k2: (1.0 if -k1 <= t <= k1 else 0.0) * numpy.exp(d ** 2 / -k2)

        sight = numpy.zeros(180)
        for (name, num, offset) in [
            ("robot", self.num_robots, 17.5),
            ("feed", self.num_feeds, 17.5),
            ("obstacle", self.num_obstacles, 17.5)
        ]:
            for i in range(num):
                sub_vector = numpy.dot(inv_rot_mat, self.model.get_body(f"{name}{i}").get_xpos() - body.get_xpos())
                distance = numpy.linalg.norm(sub_vector, ord=2) - offset - 17.5
                theta = numpy.arctan2(sub_vector[1], sub_vector[0])
                value = convertor(distance, theta, numpy.pi * 0.5, 1)
                sight_index = int(len(sight) * (0.5 - 0.5 * theta / numpy.pi) + 0.5)
                if sight[sight_index] < value:
                    sight[sight_index] = value

        return _Robot(body, act_left, act_right, sight)

    def get_feed(self, index: int):
        body = self.model.get_body(f"feed{index}")
        velocity_sensor = self.model.get_sensor(f"sensor_feed{index}_velocity")
        return _Feed(body, velocity_sensor)

    def get_obstacle(self, index):
        pos = self.model.get_geom(f"obstacle{index}").get_pos()
        return _Obstacle(pos)

    def calc_step(self) -> True:
        self.model.step()

        # Calculate Pheromone.
        for pf in self.pheromone_field:
            for _ in range(self.pheromone_iteration):
                pf.update(self.timestep / self.pheromone_iteration)

        # Stop unstable state.
        if self.model.count_raised_warning() > 0:
            print("Catch warning!")
            return False

        return True

    def render(
            self,
            loss: float,
            show_pheromone_index,
            window: miscellaneous.Window,
            camera: wrap_mjc.Camera,
            rect: (int, int, int, int) = None,
            flush: bool = True
    ):
        if not window.render(self.model, camera, rect):
            exit()
        pf = self.pheromone_field[show_pheromone_index]
        self.pheromone_panel.update(pf)

        self.model.draw_text(f"{loss}", 0, 0, (1, 1, 1))

        if flush:
            window.flush()


class Environment(optimizer.MuJoCoEnvInterface):
    def __init__(
            self,
            brain: RobotBrain,
            nest_pos: (float, float),
            robot_pos: list[(float, float)],
            obstacle_pos: list[(float, float)],
            feed_pos: list[(float, float)],
            sv: list[float],
            evaporate: list[float],
            diffusion: list[float],
            decrease: list[float],
            pheromone_field_panel_size: float,
            pheromone_field_pos: (float, float),
            pheromone_field_shape: (int, int),
            pheromone_iteration: int,
            timestep: float,
            time: int,
            show_pheromone_index: int = 0,
            window: miscellaneous.Window = None,
            camera: wrap_mjc.Camera = None
    ):
        self._world = _World(
            nest_pos,
            robot_pos,
            obstacle_pos,
            feed_pos,
            sv,
            evaporate,
            diffusion,
            decrease,
            pheromone_field_panel_size,
            pheromone_field_pos,
            pheromone_field_shape,
            pheromone_iteration,
            timestep
        )

        for _ in range(0, 5):
            self._world.calc_step()

    def calc_step(self) -> float:
        for i in range(self._world.num_robots):
            robot = self._world.get_robot(i)

        self._world.calc_step()

        # Calculate
        self.model.step()
        for pf in self.pheromone_field:
            for _ in range(self.pheromone_iteration):
                pf.update_cells(self.timestep / self.pheromone_iteration)

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
        dt_exhaustion = 0.0
        robot_pos = [r.get_pos() for r in self.robots]
        feed_pos = [f.get_pos() for f in self.feeds]
        for r, rp in zip(self.robots, robot_pos):
            pheromone_values = [pf.get_gas(rp[0], rp[1]) for pf in self.pheromone_field]
            exhaustion, secretion = r.act(pheromone_values, self.nest_pos, robot_pos, self.obstacle_pos, feed_pos)

            dt_exhaustion += exhaustion
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
        # dt_loss_obstacle_robot *= 1e12 / (len(self.obstacle_pos) * len(self.robots))
        # print(self.loss, dt_loss_feed_nest, dt_loss_feed_robot)
        self.loss += (dt_loss_feed_nest + dt_loss_feed_robot) * self.timestep  # + dt_loss_obstacle_robot

        return self.loss

    def calc(self) -> float:
        for t in range(0, int(self.time / self.timestep)):
            score = self.calc_step()
            if numpy.isinf(score):
                return score
        return self.loss

    def render(self, ):
        if self.window is not None:
            if not self.window.render(self.model, self.camera, rect):
                exit()
            self.pheromone_field[self.show_pheromone_index].update_panels()
            self.model.draw_text(f"{self.loss}", 0, 0, (1, 1, 1))
            if flush:
                self.window.flush()

    def calc_and_show(self, rect: (int, int, int, int) = None) -> float:
        for t in range(0, int(self.time / self.timestep)):
            score = self.calc_step()
            if numpy.isinf(score):
                return score
            self.render(rect)
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
        self.pheromone_iteration: int = 4
        self.show_pheromone_index: int = 0
        self.timestep: float = 0.033333
        self.time: int = 30

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
        packed.extend([struct.pack("<I", self.pheromone_iteration)])

        packed.extend([struct.pack("<d", self.timestep)])
        packed.extend([struct.pack("<I", self.time)])

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

        # フェロモンのイテレーション回数
        s = e
        e = s + 4
        self.pheromone_iteration = struct.unpack("<I", data[s:e])[0]

        # MuJoCoのタイムステップ
        s = e
        e = s + 8
        self.timestep = struct.unpack("<d", data[s:e])[0]

        # エピソードの長さ
        s = e
        e = s + 4
        self.time = struct.unpack("<I", data[s:e])[0]

        return e - offset

    def dim(self) -> int:
        return RobotBrain(None).num_dim()

    def create(self, para) -> Environment:
        l2 = numpy.linalg.norm(para, ord=2)
        brain = RobotBrain(para)
        return Environment(
            l2,
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
            self.pheromone_iteration,
            self.timestep,
            self.time,
            self.show_pheromone_index,
            None,
            None
        )

    def create_mujoco_env(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> Environment:
        l2 = numpy.linalg.norm(para, ord=2)
        brain = RobotBrain(para)
        return Environment(
            l2,
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
            self.pheromone_iteration,
            self.timestep,
            self.time,
            self.show_pheromone_index,
            window,
            camera
        )
