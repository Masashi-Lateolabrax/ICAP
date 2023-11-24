import numpy


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
