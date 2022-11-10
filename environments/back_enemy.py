import struct
import mujoco
import numpy

from environments import sensor
from studyLib import nn_tools, optimizer, wrap_mjc, miscellaneous


def gen_env(
        nest_pos: (float, float), robot_pos: [(float, float)], enemy_pos: [(float, float)],
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

    # Create Enemy
    for i, pos in enumerate(enemy_pos):
        enemy_body = worldbody.add_body(
            {"name": f"enemy{i}", "pos": f"{pos[0]} {pos[1]} 1"}
        )
        enemy_body.add_freejoint()
        enemy_body.add_geom({
            "type": "cylinder",
            "size": "10 1",
            "mass": f"{obstacle_weight}",
            "rgba": "1 0 0 1",
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

        robot_body.add_geom({
            "type": "sphere",
            "pos": "0 1 -0.3",
            "size": "0.5",
            "condim": "1"
        })

        robot_body.add_geom({
            "type": "sphere",
            "pos": "0 -1 -0.3",
            "size": "0.5",
            "condim": "1"
        })

        act.add_velocity({
            "name": f"a_robot{i}_left",
            "joint": f"joint_robot{i}_left",
            "gear": "50",
            "ctrllimited": "true",
            "ctrlrange": "-5000 5000",
            "kv": "1"
        })
        act.add_velocity({
            "name": f"a_robot{i}_right",
            "joint": f"joint_robot{i}_right",
            "gear": "50",
            "ctrllimited": "true",
            "ctrlrange": "-5000 5000",
            "kv": "1"
        })
        sen.add_velocimeter({"name": f"s_robot{i}_velocity", "site": f"site_robot{i}_body"})

    return generator.generate()


class Nest:
    def __init__(self, model: wrap_mjc.WrappedModel):
        self.geom = model.get_m_geom("nest")

    def get_pos(self):
        return self.geom.pos.copy()


class Enemy:
    def __init__(self, model: wrap_mjc.WrappedModel, number: int):
        self.body = model.get_body(f"enemy{number}")

    def get_pos(self):
        return self.body.xpos.copy()


class RobotBrain:
    def __init__(self, para):
        self.calculator = nn_tools.Calculator(6)

        self.calculator.add_layer(nn_tools.AffineLayer(10))
        self.calculator.add_layer(nn_tools.TanhLayer(10))

        self.calculator.add_layer(nn_tools.AffineLayer(2))
        self.calculator.add_layer(nn_tools.SigmoidLayer(2))

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

    def act(self, nest_pos, robot_pos: list, enemy_pos: list):
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

        enemy_omni_sensor = sensor.OmniSensor(pos, rot_mat, 50)
        for fp in enemy_pos:
            enemy_omni_sensor.sense(fp)

        input_ = numpy.concatenate([nest_direction, robot_omni_sensor.value, enemy_omni_sensor.value])
        ctrl = self.brain.calc(input_)

        if numpy.isnan(ctrl[0]) or numpy.isnan(ctrl[1]):
            print("NAN!")
        elif numpy.isinf(ctrl[0]) or numpy.isinf(ctrl[1]):
            print("INF!")

        self.left_act.ctrl = 3000 * ctrl[0]
        self.right_act.ctrl = 3000 * ctrl[1]


def evaluate(
        brain,
        nest_pos: (float, float),
        robot_pos: list,
        enemy_pos: list[(float, float)],
        enemy_weight: float,
        timestep: int,
        camera: wrap_mjc.Camera = None,
        window: miscellaneous.Window = None
) -> float:
    xml = gen_env(nest_pos, robot_pos, enemy_pos, enemy_weight)
    model = wrap_mjc.WrappedModel(xml)

    if not (camera is None):
        model.set_camera(camera)

    nest = Nest(model)
    enemies = [Enemy(model, i) for i in range(0, len(enemy_pos))]
    robots = [Robot(model, brain, i) for i in range(0, len(robot_pos))]

    loss = 0
    model.step()
    for t in range(0, timestep):
        nest_pos = nest.get_pos()
        robot_pos = [r.get_pos() for r in robots]
        enemy_pos = [f.get_pos() for f in enemies]

        # Simulate
        for r in robots:
            r.act(nest_pos, robot_pos, enemy_pos)
        model.step()

        # Calculate loss
        loss_dt = 0
        for fp in enemy_pos:
            max_dist = -float("inf")
            for rp in robot_pos:
                d = numpy.linalg.norm(rp - fp, ord=2)
                if d > max_dist:
                    max_dist = d
            enemy_dist = numpy.linalg.norm(nest_pos - fp, ord=2)
            loss_dt += 0.1 * max_dist - enemy_dist
        loss += loss_dt

        # Render MuJoCo Scene
        if not (window is None):
            if not window.render(model, camera):
                exit()
            model.draw_text(f"{loss}", 0, 0, (0, 0, 0))
            window.flush()

    return loss


class Environment(optimizer.MuJoCoEnvInterface):
    def __init__(
            self,
            nest_pos: (float, float),
            robot_pos: list[list[(float, float)]],
            enemy_pos: list[(float, float)],
            enemy_weight: float,
            timestep: int
    ):
        """
        ロボットが前方にしか進めない状態で，後方の敵を押し巣から遠くに運ぶ環境．
        ロボットはy軸の正の方向を正面として設置される．

        :param nest_pos: 巣の座標
        :param robot_pos: ロボットの座標．list[list[(float,float)]]となっており，ロボットの数が異なる複数のタスクを課すことができる．
        :param enemy_pos: 餌の座標
        :param enemy_weight: 餌の重さ
        :param timestep: stepの実行回数
        """
        self.nest_pos = nest_pos
        self.robot_pos = robot_pos
        self.enemy_pos = enemy_pos
        self.enemy_weight = enemy_weight
        self.timestep = timestep

    def dim(self) -> int:
        return RobotBrain(None).num_dim()

    def calc(self, para) -> float:
        brain = RobotBrain(para)
        total_loss = -float("inf")
        for rp in self.robot_pos:
            score = evaluate(
                brain,
                self.nest_pos,
                rp,
                self.enemy_pos,
                self.enemy_weight,
                self.timestep
            )
            if total_loss < score:
                total_loss = score
        return total_loss

    def calc_and_show(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> float:
        brain = RobotBrain(para)
        total_loss = -float("inf")
        for rp in self.robot_pos:
            score = evaluate(
                brain,
                self.nest_pos,
                rp,
                self.enemy_pos,
                self.enemy_weight,
                self.timestep,
                camera,
                window
            )
            if total_loss < score:
                total_loss = score
        return total_loss

    def save(self) -> bytes:
        packed = [struct.pack("<I", len(self.robot_pos))]  # タスクの数
        packed.extend([struct.pack("<I", len(rp)) for rp in self.robot_pos])  # タスクごとのロボットの数
        packed.extend([struct.pack("<I", len(self.enemy_pos))])  # 敵の数
        packed.extend([struct.pack("<dd", self.nest_pos[0], self.nest_pos[1])])  # 巣の座標
        packed.extend([struct.pack("<dd", x, y) for rp in self.robot_pos for (x, y) in rp])  # ロボットの座標
        packed.extend([struct.pack("<dd", x, y) for (x, y) in self.enemy_pos])  # 敵の座標
        packed.extend([struct.pack("<d", self.enemy_weight)])  # 敵の重さ
        packed.extend([struct.pack("<I", self.timestep)])  # タイムステップ
        return b"".join(packed)

    def load(self, data: bytes, offset: int = 0) -> int:
        # タスクの数
        s = offset
        e = 4
        num_task = struct.unpack("<I", data[s:e])[0]

        # タスクごとのロボットの数
        nums_robot = []
        for _i in range(0, num_task):
            s = e
            e = s + 4
            nums_robot.append(struct.unpack("<I", data[s:e])[0])

        # 敵の数
        s = e
        e = s + 4
        len_enemy_pos = struct.unpack("<I", data[s:e])[0]

        # 巣の座標
        s = e
        e = s + 16
        self.nest_pos = struct.unpack("<dd", data[s:e])

        # ロボットの座標
        self.robot_pos.clear()
        for n in nums_robot:
            pos_list = []
            for _i in range(0, n):
                s = e
                e = s + 16
                p = struct.unpack(f"<dd", data[s:e])
                pos_list.append((p[0], p[1]))
            self.robot_pos.append(pos_list)

        # 敵の座標
        self.enemy_pos = []
        for i in range(0, len_enemy_pos):
            s = e
            e = s + 16
            p = struct.unpack(f"<dd", data[s:e])
            self.enemy_pos.append((p[0], p[1]))

        # 敵の重さ
        s = e
        e = s + 8
        self.enemy_weight = struct.unpack(f"<d", data[s:e])[0]

        # タイムステップ
        s = e
        e = s + 4
        self.timestep = struct.unpack(f"<I", data[s:e])[0]

        return e - offset
