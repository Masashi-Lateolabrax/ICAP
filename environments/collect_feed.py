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

    generator.add_visual().add_headlight({
        "ambient": "0.3 0.3 0.3",
        "diffuse": "0.4 0.4 0.4",
        "specular": "0.0 0.0 0.0",
        "active": "1"
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
        feed_body = worldbody.add_body(
            name=f"feed{i}",
            pos=(fp[0], fp[1], 11)
        )
        feed_body.add_freejoint()
        feed_body.add_site({"name": f"feed{i}_center_site"})
        feed_body.add_geom({"class": "feeds"})
        sensor.add_velocimeter({
            "name": f"feed{i}_velocimeter",
            "site": f"feed{i}_center_site"
        })

    ######################################################################################################
    # Create Robots
    ######################################################################################################
    depth = 1.0
    for i, rp in enumerate(robot_pos):
        robot_body = worldbody.add_body(
            name=f"robot{i}",
            pos=(rp[0], rp[1], 10 + depth + 0.5),
            axisangle=((0, 0, 1), rp[2])
        )
        robot_body.add_freejoint()
        robot_body.add_site({"name": f"robot{i}_center_site"})
        robot_body.add_geom({"class": "robot_body"})

        right_wheel_body = robot_body.add_body(pos=(10, 0, -depth))
        right_wheel_body.add_joint({"name": f"robot{i}_right_joint", "type": "hinge", "axis": "-1 0 0"})
        right_wheel_body.add_geom({"class": "robot_wheel"})

        left_wheel_body = robot_body.add_body(pos=(-10, 0, -depth))
        left_wheel_body.add_joint({"name": f"robot{i}_left_joint", "type": "hinge", "axis": "-1 0 0"})
        left_wheel_body.add_geom({"class": "robot_wheel"})

        front_wheel_body = robot_body.add_body(pos=(0, 15, -5 + 1.5 - depth))
        front_wheel_body.add_joint({"type": "ball"})
        front_wheel_body.add_geom({"class": "robot_ball"})

        rear_wheel_body = robot_body.add_body(pos=(0, -15, -5 + 1.5 - depth))
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
        sensor.add_velocimeter({
            "name": f"robot{i}_velocimeter",
            "site": f"robot{i}_center_site"
        })

    ######################################################################################################
    # Generate XML
    ######################################################################################################
    xml = generator.generate()
    return xml


class _Obstacle:
    def __init__(self, pos):
        self.pos = pos


class _Feed:
    def __init__(self, pos, velocity):
        self.pos = pos
        self.velocity = velocity


class RobotBrain:
    def __init__(self, para):
        self._calculator = nn_tools.Calculator(105)

        self._calculator.add_layer(nn_tools.ParallelLayer(
            [
                [
                    nn_tools.FilterLayer([i for i in range(0, 100)]),

                    nn_tools.Conv1DLayer(
                        50, 4, 2,
                        nn_tools.AffineLayer(3),
                        1
                    ),
                    nn_tools.TanhLayer(150),

                    nn_tools.Conv1DLayer(
                        24, 12, 6,
                        nn_tools.AffineLayer(3),
                        0
                    ),
                    nn_tools.TanhLayer(72),

                    nn_tools.Conv1DLayer(
                        11, 12, 6,
                        nn_tools.AffineLayer(3),
                        0
                    ),
                    nn_tools.TanhLayer(33),
                ],
                [
                    nn_tools.FilterLayer([100, 101, 102, 103, 104]),
                ]
            ]
        ))

        self._calculator.add_layer(nn_tools.ParallelLayer(
            [
                [
                    nn_tools.FilterLayer([i for i in range(0, 36)]),
                    nn_tools.AffineLayer(20),
                    nn_tools.TanhLayer(20),
                    nn_tools.AffineLayer(10),
                    nn_tools.TanhLayer(10),
                    nn_tools.AffineLayer(3),
                    nn_tools.IsMaxLayer(3)
                ],
                [
                    nn_tools.FilterLayer([36, 37]),
                ]
            ]
        ))

        self._calculator.add_layer(nn_tools.AffineLayer(7))
        self._calculator.add_layer(nn_tools.TanhLayer(7))

        self._calculator.add_layer(nn_tools.AffineLayer(5))
        self._calculator.add_layer(nn_tools.TanhLayer(5))

        self._calculator.add_layer(nn_tools.AffineLayer(4))

        self._calculator.add_layer(nn_tools.ParallelLayer(
            [
                [
                    nn_tools.FilterLayer([0, 1]),
                    nn_tools.ReluLayer(2)
                ],
                [
                    nn_tools.FilterLayer([2, 3]),
                    nn_tools.SoftmaxLayer(2)
                ]
            ]
        ))

        if para is not None:
            self._calculator.load(para)

    def num_dim(self):
        return self._calculator.num_dim()

    def calc(self, array):
        return self._calculator.calc(array)


class _Robot:
    def __init__(self, pos, direction, rot, inv_rot, velocity, decision):
        self.pos = pos
        self.direction = direction
        self.rot = rot
        self.inv_rot = inv_rot
        self.velocity = velocity

        self.decision = decision


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
        self.nest_pos = numpy.array([nest_pos[0], nest_pos[1], 0])
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

        self.robot_decisions = numpy.zeros((self.num_robots, 2 + len(self.pheromone_field)))

    def calc_robot_sight(self, robot: _Robot, start, end, div):
        mat = numpy.zeros((3, 3))
        mat[2, 2] = 1.0
        step = (end - start) / div
        tmp = numpy.zeros(div)
        res = numpy.zeros(div)

        for i in range(div):
            theta = start + step * i
            mat[0, 0] = numpy.cos(theta)
            mat[0, 1] = numpy.sin(theta)
            mat[1, 0] = -numpy.sin(theta)
            mat[1, 1] = numpy.cos(theta)
            _, distance = self.model.calc_ray(
                robot.pos + robot.direction * 17.6,
                numpy.dot(mat, robot.direction)
            )
            res[i] = distance

        mask = res < 0
        numpy.multiply(0.01, res, out=tmp)
        numpy.tanh(tmp, out=res)
        numpy.subtract([1], res, out=tmp)
        tmp[mask] = 0
        return tmp

    def get_robot(self, index: int) -> _Robot:
        def calc_robot_direction(robot_body: wrap_mjc.WrappedBody):
            rot_mat = numpy.zeros(9).reshape(3, 3)
            mujoco.mju_quat2Mat(rot_mat.ravel(), robot_body.get_xquat())
            orientation = numpy.dot(rot_mat, [0, 1, 0])
            orientation[2] = 0
            distance = numpy.linalg.norm(orientation, ord=2)
            orientation /= distance
            return orientation, rot_mat, numpy.linalg.inv(rot_mat)

        body = self.model.get_body(f"robot{index}")
        velocimeter = self.model.get_sensor(f"robot{index}_velocimeter")

        pos = body.get_xpos()
        velocity = velocimeter.get_data()
        direction, rot, inv_rot = calc_robot_direction(body)

        return _Robot(pos, direction, rot, inv_rot, velocity, self.robot_decisions[index])

    def get_pheromone(self, x, y):
        return [field.get_gas(x, y) for field in self.pheromone_field]

    def get_feed(self, index: int):
        body = self.model.get_body(f"feed{index}")
        velocimeter = self.model.get_sensor(f"feed{index}_velocimeter")
        return _Feed(body.get_xpos(), velocimeter.get_data())

    def get_obstacle(self, index):
        pos = self.model.get_geom(f"obstacle{index}").get_pos()
        return _Obstacle(pos)

    def calc_step(self) -> True:
        # Make robots move
        for ri, rd in enumerate(self.robot_decisions):
            act_left = self.model.get_act(f"robot{ri}_left_act")
            act_right = self.model.get_act(f"robot{ri}_right_act")
            pos = self.model.get_body(f"robot{ri}").get_xpos()

            act_left.ctrl = 350 * rd[0]
            act_right.ctrl = 350 * rd[1]
            self.secretion(pos, rd[2:])

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

    def secretion(self, pos, secretion):
        for i, s in enumerate(secretion):
            self.pheromone_field[i].add_liquid(pos[0], pos[1], s)


class RobotSightViewer:
    def __init__(self, model: wrap_mjc.WrappedModel):
        self.cells = []
        offset = (700, 600)
        w = 12
        h = 300
        for i in range(100):
            geom = model.add_deco_geom(mujoco.mjtGeom.mjGEOM_PLANE)
            self.cells.append(geom)
            geom.set_size((w, h, 0.05))
            geom.set_pos((w * i + offset[0], offset[1], 0))

    def update(self, sight):
        for i, s in enumerate(sight):
            self.cells[i].set_rgba((s, s, s, 1))


class Environment(optimizer.MuJoCoEnvInterface):
    def __init__(
            self,
            l2: float,
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
            thinking_interval,
            show_pheromone_index: int = 0,
            window: miscellaneous.Window = None,
            camera: wrap_mjc.Camera = None
    ):
        self._loss = l2

        self._thinking_interval = thinking_interval
        self._thinking_counter = 0

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

        self.robot_brain = brain

        self.time = time
        self.show_pheromone_index = show_pheromone_index
        self.window = window
        self.camera = camera

        for _ in range(0, 5):
            self._world.calc_step()

    def calc_step(self) -> float:
        if not self._world.calc_step():
            return float("inf")

        if self._thinking_counter <= 0:
            for i in range(self._world.num_robots):
                robot = self._world.get_robot(i)

                robot_sight = self._world.calc_robot_sight(robot, -numpy.pi * 0.55, numpy.pi * 0.55, 100)
                pheromone = self._world.get_pheromone(robot.pos[0], robot.pos[1])

                nest_vec = numpy.dot(robot.inv_rot, self._world.nest_pos - robot.pos)
                nest_dist = numpy.linalg.norm(nest_vec, ord=2)
                nest_vec /= nest_dist if nest_dist > 0.0 else 0.0
                decreased_nest_dist = (1.0 / (nest_dist + 1.0)) ** 2
                sensed_nest = [nest_vec[0], nest_vec[1], decreased_nest_dist]

                robot.decision[:] = self.robot_brain.calc(numpy.concatenate([robot_sight, sensed_nest, pheromone]))

        if self._thinking_counter <= 0:
            self._thinking_counter = self._thinking_interval
        self._thinking_counter -= 1

        for i in range(self._world.num_obstacles):
            o = self._world.get_obstacle(i)
            self._world.secretion(o.pos, [2, 0])

        # Calculate loss
        feed_range = 15000.0
        obstacle_range = 200.0
        dt_loss_feed_nest = 0.0
        dt_loss_feed_robot = 0.0
        dt_loss_obstacle_robot = 0.0
        feed_speed = numpy.zeros(3)
        for i in range(self._world.num_feeds):
            feed = self._world.get_feed(i)

            feed_nest_vector = self._world.nest_pos - feed.pos
            feed_nest_distance = numpy.linalg.norm(feed_nest_vector, ord=2)
            normed_feed_nest_vector = feed_nest_vector / feed_nest_distance
            feed_speed[:] = feed.velocity
            feed_speed[2] = 0

            dt_loss_feed_nest -= numpy.dot(feed_speed, normed_feed_nest_vector)

            for j in range(self._world.num_robots):
                robot = self._world.get_robot(j)
                feed_robot_vector = (robot.pos - feed.pos)[0:2]
                feed_robot_distance = numpy.sum(feed_robot_vector ** 2)
                dt_loss_feed_robot -= numpy.exp(-feed_robot_distance / feed_range)

        for i in range(self._world.num_obstacles):
            obstacle = self._world.get_obstacle(i)
            for j in range(self._world.num_robots):
                robot = self._world.get_robot(j)
                obstacle_robot_vector = robot.pos - obstacle.pos
                obstacle_robot_distance = numpy.sum(obstacle_robot_vector ** 2)
                dt_loss_obstacle_robot += numpy.exp(-obstacle_robot_distance / obstacle_range)

        dt_loss_feed_nest *= 1e0 / self._world.num_feeds
        dt_loss_feed_robot *= 1e11 / (self._world.num_feeds * self._world.num_robots)
        dt_loss_obstacle_robot *= 1e32 / (self._world.num_obstacles * self._world.num_robots)
        self._loss += (dt_loss_feed_nest + dt_loss_feed_robot + dt_loss_obstacle_robot) * self._world.timestep

        return self._loss

    def calc(self) -> float:
        for t in range(0, int(self.time / self._world.timestep)):
            score = self.calc_step()
            if numpy.isinf(score):
                return score
        return self._loss

    def render(self, rect: (int, int, int, int) = None, flush: bool = True) -> float:
        self._world.render(
            self._loss,
            self.show_pheromone_index,
            self.window,
            self.camera,
            rect,
            flush
        )
        return 0.0

    def calc_and_show(self, rect: (int, int, int, int) = None) -> float:
        for t in range(0, int(self.time / self._world.timestep)):
            score = self.calc_step()
            if numpy.isinf(score):
                return score
            self.render(rect)
        return self._loss


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
        self.think_interval: int = 0

    def save(self):
        import struct

        packed = [struct.pack("<dd", self.nest_pos[0], self.nest_pos[1])]

        packed.extend([struct.pack("<I", len(self.robot_pos))])
        packed.extend([struct.pack("<ddd", p[0], p[1], p[2]) for p in self.robot_pos])

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
        packed.extend([struct.pack("<I", self.think_interval)])

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
            e = s + 24
            rp = struct.unpack("<ddd", data[s:e])[0:3]
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

        # ロボットの演算の間隔
        s = e
        e = s + 4
        self.think_interval = struct.unpack("<I", data[s:e])[0]

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
            self.think_interval,
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
            self.think_interval,
            self.show_pheromone_index,
            window,
            camera
        )
