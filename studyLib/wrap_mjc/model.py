import math
import mujoco
import numpy


class Camera:
    def __init__(
            self,
            lookat_xyz: (float, float, float),
            distance: float,
            horizontal_angle: float,
            vertical_angle: float
    ):
        self.look_at = lookat_xyz
        self.distance = distance
        self.horizontal_angle = horizontal_angle
        self.vertical_angle = vertical_angle


class DecoGeom(mujoco.MjvGeom):
    def __init__(self, geom_type: mujoco.mjtGeom):
        super().__init__()
        mujoco.mjv_initGeom(
            self, geom_type,
            numpy.ones(3), numpy.ones(3), numpy.eye(3, 3).ravel(),
            numpy.ones(4).astype(numpy.float32)
        )

    def copy_to(self, destination: mujoco.MjvGeom):
        mujoco.mjv_initGeom(
            destination,
            self.type,
            self.size,
            self.pos,
            self.mat,
            self.rgba,
        )

    def get_type(self) -> int:
        return self.type

    def get_size(self) -> numpy.ndarray:
        return self.size.copy()

    def set_size(self, size):
        self.size = numpy.array(size).reshape((3,))

    def get_pos(self) -> numpy.ndarray:
        return self.pos.copy()

    def set_pos(self, pos):
        self.pos = numpy.array(pos).reshape((3,))

    def get_quat(self) -> numpy.ndarray:
        quat = numpy.zeros(4)
        mujoco.mju_mat2Quat(quat, self.mat.ravel())
        return quat

    def set_quat(self, axis, theta):
        quat = numpy.zeros(4)
        quat[0] = math.cos(theta * 0.5)
        quat[1] = axis[0] * math.sin(theta * 0.5)
        quat[2] = axis[1] * math.sin(theta * 0.5)
        quat[3] = axis[2] * math.sin(theta * 0.5)
        mat = numpy.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        self.mat = mat.reshape(3, 3)

    def get_rgba(self) -> numpy.ndarray:
        return self.rgba.copy()

    def set_rgba(self, rgba):
        self.rgba = numpy.array(rgba).reshape((4,))


class WrappedModel:
    def __init__(self, xml: str):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        self.deco_geoms: list[DecoGeom] = []
        self._scn: mujoco.MjvScene = None
        self._ctx: mujoco.MjrContext = None

    def add_deco_geom(self, geom_type: mujoco.mjtGeom) -> DecoGeom:
        self.deco_geoms.append(DecoGeom(geom_type))
        return self.deco_geoms[-1]

    def remove_deco_geom(self, geom: DecoGeom):
        self.deco_geoms.remove(geom)

    def step(self):
        self.coolback()
        mujoco.mj_step(self.model, self.data)

    def set_camera(self, cam: Camera):
        self.camera.lookat[0] = cam.look_at[0]
        self.camera.lookat[1] = cam.look_at[1]
        self.camera.lookat[2] = cam.look_at[2]
        self.camera.distance = cam.distance
        self.camera.elevation = -cam.vertical_angle
        self.camera.azimuth = cam.horizontal_angle

    def get_ctx(self) -> mujoco.MjrContext:
        if self._ctx is None:
            self._ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        return self._ctx

    def get_scene(self, cam: Camera = None):
        if self._scn is None:
            self._scn = mujoco.MjvScene(self.model, maxgeom=self.model.ngeom + len(self.deco_geoms) + 1)

        if self._scn.maxgeom != self.model.ngeom + len(self.deco_geoms) + 1:
            self._scn = mujoco.MjvScene(self.model, maxgeom=self.model.ngeom + len(self.deco_geoms) + 1)

        if not (cam is None):
            self.set_camera(cam)

        mujoco.mjv_updateScene(
            self.model, self.data,
            mujoco.MjvOption(), None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL, self._scn
        )

        for dest, src in zip(self._scn.geoms[self.model.ngeom:], self.deco_geoms):
            src.copy_to(dest)
        self._scn.ngeom = self.model.ngeom + len(self.deco_geoms)

        return self._scn

    def draw_text(self, text: str, x: float, y: float, rgb: (float, float, float)):
        ctx = self.get_ctx()
        mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, text, ctx, x, y, rgb[0], rgb[1], rgb[2])

    def get_names(self):
        split = self.model.names.split(b"\x00")
        return [name.decode("utf-8") for name in split if name != b""]

    def get_body(self, name: str):
        return self.data.body(name)

    def get_num_geom(self):
        if self._scn.ngeom is None:
            return self.model.ngeom
        return self._scn.ngeom

    def get_d_geom(self, name: str):
        return self.data.geom(name)

    def get_m_geom(self, name: str):
        return self.model.geom(name)

    def get_act(self, act_name: str):
        return self.data.actuator(act_name)

    def set_act_ctrl(self, act_name: str, value):
        self.data.actuator(act_name).ctrl = value

    def get_sensor(self, sen_name: str):
        return self.data.sensor(sen_name)

    # コールバック関数のタイポ．ループごとに実行される処理を記述．別に書かなくてもＯＫ．
    def coolback(self):
        pass
