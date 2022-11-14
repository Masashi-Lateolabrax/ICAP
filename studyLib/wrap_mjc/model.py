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
            self.mat.ravel(),
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


class WrappedBody:
    def __init__(self, model_body, data_body):
        self._model_body = model_body
        self._data_body = data_body

    def get_cacc(self):
        return self._data_body.cacc

    def get_cfrc_ext(self):
        return self._data_body.cfrc_ext

    def get_cfrc_int(self):
        return self._data_body.cfrc_int

    def get_cinert(self):
        return self._data_body.cinert

    def get_crb(self):
        return self._data_body.crb

    def get_cvel(self):
        return self._data_body.cvel

    def get_id(self):
        return self._data_body.id

    def get_name(self):
        return self._data_body.name

    def get_subtree_angmom(self):
        return self._data_body.subtree_angmom

    def get_subtree_com(self):
        return self._data_body.subtree_com

    def get_subtree_linvel(self):
        return self._data_body.subtree_linvel

    def get_xfrc_applied(self):
        return self._data_body.xfrc_applied

    def get_ximat(self):
        return self._data_body.ximat

    def get_xipos(self):
        return self._data_body.xipos

    def get_xmat(self):
        return self._data_body.xmat

    def get_xpos(self) -> numpy.ndarray:
        return self._data_body.xpos

    def get_xquat(self) -> numpy.ndarray:
        return self._data_body.xquat

    def get_dofadr(self):
        return self._model_body.dofadr

    def get_dofnum(self):
        return self._model_body.dofnum

    def get_geomadr(self):
        return self._model_body.geomadr

    def get_geomnum(self):
        return self._model_body.geomnum

    def get_inertia(self):
        return self._model_body.inertia

    def get_invweight0(self):
        return self._model_body.invweight0

    def get_ipos(self):
        return self._model_body.ipos

    def get_iquat(self):
        return self._model_body.iquat

    def get_jntadr(self):
        return self._model_body.jntadr

    def get_jntnum(self):
        return self._model_body.jntnum

    def get_mass(self) -> numpy.ndarray:
        """
        Bodyの重さを返します．単位はkgです．
        :return: 重さが格納された配列
        """
        return self._model_body.mass

    def get_parentid(self):
        return self._model_body.parentid

    def get_sameframe(self):
        return self._model_body.sameframe

    def get_simple(self):
        return self._model_body.simple

    def get_subtreemass(self):
        return self._model_body.subtreemass


class WrappedGeom:
    def __init__(self, model_geom, data_geom):
        self.model_geom = model_geom,
        self.data_geom = data_geom

    def get_id(self):
        return self.data_geom.id

    def get_name(self):
        return self.data_geom.name

    def get_xmat(self):
        return self.data_geom.xmat

    def get_xpos(self) -> numpy.ndarray:
        return self.data_geom.xpos

    def get_conaffinity(self):
        return self.model_geom.conaffinity

    def get_condim(self):
        return self.model_geom.condim

    def get_contype(self):
        return self.model_geom.contype

    def get_friction(self):
        return self.model_geom.friction

    def get_gap(self):
        return self.model_geom.gap

    def get_group(self):
        return self.model_geom.group

    def get_margin(self):
        return self.model_geom.margin

    def get_priority(self):
        return self.model_geom.priority

    def get_rbound(self):
        return self.model_geom.rbound

    def get_rgba(self):
        return self.model_geom.rgba

    def get_sameframe(self):
        return self.model_geom.sameframe

    def get_size(self):
        return self.model_geom.size

    def get_solimp(self):
        return self.model_geom.solimp

    def get_solmix(self):
        return self.model_geom.solmix

    def get_solref(self):
        return self.model_geom.solref

    def get_type(self):
        return self.model_geom.type


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
        mujoco.mjr_text(mujoco.mjtFont.mjFONT_BIG, text, ctx, x, y, rgb[0], rgb[1], rgb[2])

    def get_names(self) -> list[str]:
        split = self.model.names.split(b"\x00")
        return [name.decode("utf-8") for name in split if name != b""]

    def get_body(self, name: str) -> WrappedBody:
        return WrappedBody(self.model.body(name), self.data.body(name))

    def get_num_geom(self) -> int:
        if self._scn.ngeom is None:
            return self.model.ngeom
        return self._scn.ngeom

    def get_geom(self, name: str):
        return WrappedGeom(self.model.geom(name), self.data.geom(name))

    def get_act(self, act_name: str):
        return self.data.actuator(act_name)

    def set_act_ctrl(self, act_name: str, value):
        self.data.actuator(act_name).ctrl = value

    def get_sensor(self, sen_name: str):
        return self.data.sensor(sen_name)

    # コールバック関数のタイポ．ループごとに実行される処理を記述．別に書かなくてもＯＫ．
    def coolback(self):
        pass
