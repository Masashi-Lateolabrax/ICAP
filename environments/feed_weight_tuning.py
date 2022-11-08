import struct

import glfw
import mujoco
import numpy

from environments import sensor
from studyLib import nn_tools, optimizer, wrap_mjc


def gen_env(
        nest_pos: (float, float), robot_pos: [(float, float)], feed_pos: [(float, float)],
        obstacle_weight: float
):
    generator = wrap_mjc.MuJoCoXMLGenerator("co-behavior")

    generator.add_option({"timestep": 0.03333})
    generator.add_asset().add_texture({
        "type": "skybox",
        "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7",
        "rgb2": "0 0 0",
        "width": "512",
        "height": "512",
    })

    worldbody = generator.get_body()

    # Create Ground
    worldbody.add_geom({
        "type": "plane",
        "size": "0 0 0.05",
        "rgba": "1.0 1.0 1.0 0.5",
        "condim": "1",
        "priority": "0"
    })

    # Create Nest
    worldbody.add_geom({
        "name": "nest",
        "type": "cylinder",
        "size": "5 0.5",
        "rgba": "0 1 0 1",
        "pos": f"{nest_pos[0]} {nest_pos[1]} -2"
    })

    # Create Feed
    for i, pos in enumerate(feed_pos):
        feed_body = worldbody.add_body(
            {"name": f"feed{i}", "pos": f"{pos[0]} {pos[1]} 1"}
        )
        feed_body.add_freejoint()
        feed_body.add_geom({
            "type": "cylinder",
            "size": "10 1",
            "mass": f"{obstacle_weight}",
            "rgba": "0 0 1 1",
            "condim": "3",
            "priority": "1",
            "friction": "1 0.005 0.0001"
        })

    act = generator.add_actuator()
    sen = generator.add_sensor()

    # Create Robots
    for i, pos in enumerate(robot_pos):
        robot_body = worldbody.add_body({
            "name": f"robot{i}",
            "pos": f"{pos[0]} {pos[1]} 1",
            "axisangle": f"0 0 1 0"
        })

        robot_body.add_freejoint()
        robot_body.add_geom({
            "type": "cylinder",
            "size": "1.75 0.5",
            "mass": "5000",
            "rgba": "1 1 0 0.3",
        })
        robot_body.add_site({
            "name": f"site_robot{i}_body",
            "rgba": "0 0 0 0",
        })

        right_wheel_body = robot_body.add_body({"pos": "1 0 -0.3"})
        right_wheel_body.add_joint({"name": f"joint_robot{i}_right", "type": "hinge", "axis": "-1 0 0"})
        right_wheel_body.add_geom({
            "type": "cylinder",
            "size": "0.5 0.5",
            "axisangle": "0 1 0 90",
            "condim": "6",
            "priority": "1",
            "friction": "1.0 0.005 0.00001"
        })

        left_wheel_body = robot_body.add_body({"pos": "-1 0 -0.3"})
        left_wheel_body.add_joint({"name": f"joint_robot{i}_left", "type": "hinge", "axis": "-1 0 0"})
        left_wheel_body.add_geom({
            "type": "cylinder",
            "size": "0.5 0.5",
            "axisangle": "0 1 0 90",
            "condim": "6",
            "priority": "1",
            "friction": "1.0 0.005 0.00001"
        })

        front_wheel_body = robot_body.add_body({"pos": "0 1 -0.3"})
        front_wheel_body.add_joint({"type": "ball"})
        front_wheel_body.add_geom({
            "type": "sphere",
            "size": "0.5",
            "condim": "1"
        })

        back_wheel_body = robot_body.add_body({"pos": "0 -1 -0.3"})
        back_wheel_body.add_joint({"type": "ball"})
        back_wheel_body.add_geom({
            "type": "sphere",
            "size": "0.5",
            "condim": "1"
        })

        act.add_velocity({
            "name": f"a_robot{i}_left",
            "joint": f"joint_robot{i}_left",
            "gear": "50",
            "ctrllimited": "true",
            "ctrlrange": "-1000 1000",
            "kv": "1"
        })
        act.add_velocity({
            "name": f"a_robot{i}_right",
            "joint": f"joint_robot{i}_right",
            "gear": "50",
            "ctrllimited": "true",
            "ctrlrange": "-1000 1000",
            "kv": "1"
        })
        sen.add_velocimeter({"name": f"s_robot{i}_velocity", "site": f"site_robot{i}_body"})

    return generator.generate()


class Nest:
    def __init__(self, model: wrap_mjc.WrappedModel):
        self.geom = model.get_m_geom("nest")

    def get_pos(self):
        return self.geom.pos.copy()


class Feed:
    def __init__(self, model: wrap_mjc.WrappedModel, number: int):
        self.body = model.get_body(f"feed{number}")

    def get_pos(self):
        return self.body.xpos.copy()


class RobotBrain:
    def __init__(self, para):
        self.calculator = nn_tools.Calculator(6)

        self.calculator.add_layer(nn_tools.AffineLayer(10))
        self.calculator.add_layer(nn_tools.TanhLayer(10))

        self.calculator.add_layer(nn_tools.AffineLayer(2))
        self.calculator.add_layer(nn_tools.TanhLayer(2))

        if not (para is None):
            self.calculator.load(para)

    def num_dim(self):
        return self.calculator.num_dim()

    def calc(self, array):
        return self.calculator.calc(array)


class Robot:
    def __init__(self, model: wrap_mjc.WrappedModel, brain: RobotBrain, number: int):
        self.body = model.get_body(f"robot{number}")
        self.brain = brain
        self.left_act = model.get_act(f"a_robot{number}_left")
        self.right_act = model.get_act(f"a_robot{number}_right")

    def get_pos(self):
        return self.body.xpos.copy()

    def get_orientation(self):
        rot_mat = numpy.zeros(9).reshape(3, 3)
        mujoco.mju_quat2Mat(rot_mat.ravel(), self.body.xquat)
        return rot_mat

    def get_direction(self):
        return numpy.dot(self.get_orientation(), [0.0, 1.0, 0.0])

    def act(self, nest_pos, robot_pos: list, feed_pos: list):
        pos = self.get_pos()
        rot_mat = self.get_orientation()

        ref_nest_pos = (numpy.array(nest_pos) - pos)[:2]
        nest_dist = numpy.linalg.norm(ref_nest_pos, ord=2)
        if nest_dist > 0.0:
            nest_direction = ref_nest_pos / nest_dist
        else:
            nest_direction = numpy.zeros(2)

        robot_omni_sensor = sensor.OmniSensor(pos, rot_mat, 15)
        for rp in robot_pos:
            robot_omni_sensor.sense(rp)

        feed_omni_sensor = sensor.OmniSensor(pos, rot_mat, 50)
        for fp in feed_pos:
            feed_omni_sensor.sense(fp)

        input_ = numpy.concatenate([nest_direction, robot_omni_sensor.value, feed_omni_sensor.value])
        ctrl = self.brain.calc(input_)
        self.left_act.ctrl = 1000 * ctrl[0]
        self.right_act.ctrl = 1000 * ctrl[1]


def evaluate(
        brain,
        nest_pos: (float, float),
        robot_pos: list[(float, float)],
        feed_pos: list[(float, float)],
        feed_weight: float,
        timestep: int,
        camera: (float, float, float) = None,
        window=None
) -> float:
    xml = gen_env(nest_pos, robot_pos, feed_pos, feed_weight)
    model = wrap_mjc.WrappedModel(xml)

    if not (camera is None):
        model.set_camera((0, 0, 0), camera[0], camera[1], camera[2])

    nest = Nest(model)
    feeds = [Feed(model, i) for i in range(0, len(feed_pos))]
    robots = [Robot(model, brain, i) for i in range(0, len(robot_pos))]

    loss = 0
    model.step()
    for t in range(0, timestep):
        nest_pos = nest.get_pos()
        robot_pos = [r.get_pos() for r in robots]
        feed_pos = [f.get_pos() for f in feeds]

        # Simulate
        for r in robots:
            r.act(nest_pos, robot_pos, feed_pos)
        model.step()

        # Calculate loss
        loss_dt = 0
        for fp in feed_pos:
            max_dist = -float("inf")
            for rp in robot_pos:
                d = numpy.linalg.norm(rp - fp, ord=2)
                if d > max_dist:
                    max_dist = d
            feed_dist = numpy.linalg.norm(nest_pos - fp, ord=2)
            loss_dt += 0.1 * max_dist - feed_dist
        loss += loss_dt

        # Render MuJoCo Scene
        if not (window is None):
            if glfw.window_should_close(window):
                exit()
            model.update_scene()
            model.render_scene(window)
    return loss


class Environment(optimizer.EnvInterface):
    def __init__(
            self,
            nest_pos: (float, float),
            robot_pos: list[(float, float)],
            feed_pos: list[(float, float)],
            feed_weight: float,
            timestep: int,
            camera: (float, float, float) = None,
            window=None
    ):
        self.nest_pos = nest_pos
        self.robot_pos = robot_pos
        self.feed_pos = feed_pos
        self.feed_weight = feed_weight
        self.timestep = timestep
        self.camera = camera
        self.window = window

    def dim(self) -> int:
        return RobotBrain(None).num_dim()

    def calc(self, para) -> float:
        brain = RobotBrain(para)
        return evaluate(
            brain,
            self.nest_pos,
            self.robot_pos,
            self.feed_pos,
            self.feed_weight,
            self.timestep,
            self.camera,
            self.window
        )

    def save(self) -> bytes:
        packed = [struct.pack("<I", len(self.robot_pos))]
        packed.extend([struct.pack("<I", len(self.feed_pos))])
        packed.extend([struct.pack("<d", x) for x in self.nest_pos])
        packed.extend([struct.pack("<d", y) for x in self.robot_pos for y in x])
        packed.extend([struct.pack("<d", y) for x in self.feed_pos for y in x])
        packed.extend([struct.pack("<d", self.feed_weight)])
        packed.extend([struct.pack("<I", self.timestep)])
        return b"".join(packed)

    def load(self, data: bytes, offset: int = 0) -> int:
        s = offset
        e = 4
        len_robot_pos = struct.unpack("<I", data[s:e])[0]

        s = e
        e = s + 4
        len_feed_pos = struct.unpack("<I", data[s:e])[0]

        s = e
        e = s + 16
        self.nest_pos = struct.unpack("<2d", data[s:e])

        self.robot_pos = []
        for i in range(0, len_robot_pos):
            s = e
            e = s + 16
            self.robot_pos.append(struct.unpack(f"<2d", data[s:e]))

        self.feed_pos = []
        for i in range(0, len_feed_pos):
            s = e
            e = s + 16
            self.feed_pos.append(struct.unpack(f"<2d", data[s:e]))

        s = e
        e = s + 8
        self.feed_weight = struct.unpack(f"<d", data[s:e])[0]

        s = e
        e = s + 4
        self.timestep = struct.unpack(f"<I", data[s:e])[0]

        return e - offset
