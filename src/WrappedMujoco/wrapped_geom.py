import numpy
import mujoco


class WrappedGeom:
    def __init__(self, model_geom, data_geom):
        self.model_geom: mujoco.MjModel = model_geom
        self.data_geom: mujoco.MjvGeom = data_geom

    def get_id(self):
        return self.data_geom.id

    def get_name(self):
        return self.model_geom.name

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

    def get_pos(self):
        return self.model_geom.pos

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
