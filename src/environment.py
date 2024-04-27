import pickle

import mujoco
import numpy as np

import utils
from optimizer import EnvCreator, EnvInterface
from brain import NeuralNetwork
from distance_measure import DistanceMeasure
from robot import Robot

import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import WorldBody, Body, body
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Actuator, actuator
from mujoco_xml_generator import Asset, asset


def gen_xml(
        bots: list[tuple[float, float, float]],
        goals: list[tuple[float, float, float]],
        timestep: float
) -> str:
    generator = mjc_gen.Generator().add_children([
        Option(timestep=timestep, impratio=10, noslip_iterations=5),
        Visual().add_children([
            visual.Global(offwidth=500, offheight=500)
        ]),
        Asset().add_children([
            asset.Texture(
                name="simple_checker", type_=mjc_cmn.TextureType.TWO_DiM, builtin=mjc_cmn.TextureBuiltinType.CHECKER,
                width=100, height=100, rgb1=(1., 1., 1.), rgb2=(0.7, 0.7, 0.7)
            ),
            asset.Material(
                name="ground", texture="simple_checker", texrepeat=(10, 10)
            )
        ])
    ])

    worldbody = WorldBody().add_children([
        body.Geom(
            type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(5, 5, 1), material="ground",
        ),
    ])
    act = Actuator()

    for i, p in enumerate(goals):
        worldbody.add_children([
            body.Geom(
                name=f"goal{i}", type_=mjc_cmn.GeomType.CYLINDER, pos=(p[0], p[1], 0.025), size=(0.4, 0.025),
                rgba=(0, 1, 0, 1), conaffinity=2, contype=2,
            )
        ])

    for i, p in enumerate(bots):
        worldbody.add_children([
            Body(
                name=f"bot{i}.body", pos=(p[0], p[1], 0.06), orientation=mjc_cmn.Orientation.AxisAngle(0, 0, 1, p[2])
            ).add_children([
                body.Geom(
                    name=f"bot{i}.geom", type_=mjc_cmn.GeomType.CYLINDER,
                    size=(0.3, 0.05), rgba=(1, 1, 0, 0.5), mass=30e3,
                ),

                body.Joint(name=f"bot{i}.joint.slide_x", type_=mjc_cmn.JointType.SLIDE, axis=(1, 0, 0)),
                body.Joint(name=f"bot{i}.joint.slide_y", type_=mjc_cmn.JointType.SLIDE, axis=(0, 1, 0)),
                body.Joint(name=f"bot{i}.joint.hinge", type_=mjc_cmn.JointType.HINGE, axis=(0, 0, 1)),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.05,), rgba=(1, 0, 0, 1), pos=(0, 0.2, 0.051),
                ),
                body.Site(name=f"bot{i}.site.center", type_=mjc_cmn.GeomType.SPHERE, size=(0.05,)),

                body.Camera(
                    f"bot{i}.camera", pos=(0, 0.31, 0),
                    orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90)
                ),
                body.Camera(
                    f"bot{i}.camera_top", pos=(0, 0, 2),
                )
            ])
        ])
        act.add_children([
            actuator.Position(
                name=f"bot{i}.act.pos_x", joint=f"bot{i}.joint.slide_x",
                kp=30000000, kv=1000000
            ),
            actuator.Position(
                name=f"bot{i}.act.pos_y", joint=f"bot{i}.joint.slide_y",
                kp=30000000, kv=1000000
            ),
            actuator.Position(
                name=f"bot{i}.act.rot", joint=f"bot{i}.joint.hinge",
                kp=1000000, kv=100000
            ),
        ])

    generator.add_children([
        worldbody,
        act
    ])

    xml = generator.build()
    return xml


class _MuJoCoProcess:
    def __init__(
            self,
            brain,
            bot_pos: list[list[tuple[float, float, float]]],
            goal_pos: list[list[tuple[float, float, float]]],
            timestep: float,
    ):
        self._brain = brain
        self._measure = DistanceMeasure(64)

        self._bot_pos = bot_pos
        self._goal_pos = goal_pos

        self._num_bot = len(self._bot_pos[0])
        self._num_goal = len(self._goal_pos[0])
        self.timestep = timestep

        self.m: mujoco.MjModel = None
        self.d: mujoco.MjData = None
        self.bots: list[Robot] = None

    def init_mujoco(self, try_count: int):
        xml = gen_xml(
            self._bot_pos[try_count],
            self._goal_pos[try_count],
            self.timestep
        )

        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.bots = [Robot(self.m, self.d, i, self._brain) for i in range(self._num_bot)]

    def calc_step(self):
        mujoco.mj_step(self.m, self.d)
        for bot in self.bots:
            bot.exec(self.m, self.d, self._measure)

        evaluation = 0
        for gi in range(self._num_goal):
            goal_geom_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"goal{gi}")
            goal_geom = self.d.geom(goal_geom_id)
            min_d = float("inf")
            for ri in range(self._num_bot):
                bot_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"bot{ri}.body")
                bot_body = self.d.body(bot_body_id)
                d = np.linalg.norm(bot_body.xpos - goal_geom.xpos)
                if d < min_d:
                    min_d = d
            evaluation += min_d

        return evaluation


class Environment(EnvInterface):
    def __init__(
            self,
            bot_pos: list[list[tuple[float, float, float]]],
            goal_pos: list[list[tuple[float, float, float]]],
            brain,
            timestep: float
    ):
        self._try_count = len(bot_pos)
        self.mujoco = _MuJoCoProcess(brain, bot_pos, goal_pos, timestep)
        self._direction_buf = np.zeros((3, 1), dtype=np.float64)

    def calc_step(self) -> float:
        return self.mujoco.calc_step()

    def calc(self) -> float:
        evaluations = np.zeros(self._try_count)
        for t in range(self._try_count):
            self.mujoco.init_mujoco(t)
            for _ in range(int(15 / self.mujoco.timestep + 0.5)):
                evaluations[t] += self.calc_step()
        return np.median(evaluations)


class ECreator(EnvCreator):
    def __init__(self, num_bot, num_goal, try_count, timestep):
        import random
        self.num_bot = num_bot
        self.num_goal = num_goal
        self.try_count = try_count
        self.timestep = timestep

        self._brain = NeuralNetwork()

        self.bot_pos = [
            [(0, -5, 360 * random.random())] for _ in range(self.try_count)
        ]
        self.goal_pos = [
            [(0, 5, 0)] for _ in range(self.try_count)
        ]

    def save(self) -> bytes:
        return pickle.dumps(self)

    def load(self, byte_data: bytes) -> int:
        new_instance = pickle.loads(byte_data)
        self.__dict__.update(new_instance.__dict__)
        return len(byte_data)

    def create(self, para) -> Environment:
        self._brain.load_para(para)
        return Environment(self.bot_pos, self.goal_pos, self._brain, self.timestep)
