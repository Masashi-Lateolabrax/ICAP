import mujoco
import numpy

from . import *


class WrappedModel:
    def __init__(self, xml: str):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.deco_geoms: list[DecoGeom] = []
        self.camera = mujoco.MjvCamera()
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

    def set_global_camera(self, cam: Camera):
        self.camera.lookat[0] = cam.look_at[0]
        self.camera.lookat[1] = cam.look_at[1]
        self.camera.lookat[2] = cam.look_at[2]
        self.camera.distance = cam.distance
        self.camera.elevation = -cam.vertical_angle
        self.camera.azimuth = cam.horizontal_angle

    def get_camera(self, name: str) -> WrappedCamera:
        raw_camera = self.model.camera(name)
        return WrappedCamera(raw_camera)

    def count_raised_warning(self) -> int:
        s = 0
        for i in self.data.warning.number:
            s += i
        return s

    def get_ctx(self) -> mujoco.MjrContext:
        if self._ctx is None:
            self._ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100.value)
        return self._ctx

    def get_scene(self, cam: Camera = None):
        num_viewable_geom = self.model.ngeom + self.model.ntendon * 32 + len(self.deco_geoms)

        if self._scn is None:
            self._scn = mujoco.MjvScene(self.model, maxgeom=num_viewable_geom)
        elif self._scn.maxgeom != num_viewable_geom:
            self._scn = mujoco.MjvScene(self.model, maxgeom=num_viewable_geom)

        if cam is not None:
            self.set_global_camera(cam)

        mujoco.mjv_updateScene(
            self.model, self.data,
            mujoco.MjvOption(), None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL, self._scn
        )

        for dest, src in zip(reversed(self._scn.geoms), self.deco_geoms):
            src.copy_to(dest)
        self._scn.ngeom = num_viewable_geom

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

    def get_sensor(self, sen_name: str) -> WrappedSensor:
        return WrappedSensor(self.data.sensor(sen_name))

    # コールバック関数のタイポ．ループごとに実行される処理を記述．別に書かなくてもＯＫ．
    def coolback(self):
        pass

    def calc_ray(
            self,
            start_point: (float, float, float),
            vector: (float, float, float),
            exclude_id: int = -1
    ) -> (str, float):
        i = numpy.zeros((1, 1), dtype=numpy.int32)
        pnt = numpy.array(start_point).reshape((3, 1))
        vec = numpy.array(vector).reshape((3, 1))
        distance = mujoco.mj_ray(
            self.model, self.data,
            pnt=pnt, vec=vec,
            geomgroup=None,
            flg_static=1,
            bodyexclude=exclude_id,
            geomid=i
        )
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i[0, 0]) if distance >= 0.0 else ""
        return name, distance
