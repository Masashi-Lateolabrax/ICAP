import mujoco
from mujoco._specs import MjsSite

from ..environment import add_site
from .cell import PheromoneFieldCell


class PheromoneFieldCellSpec(MjsSite):
    def __new__(cls, site: MjsSite, name: str, index_x: int, index_y: int):
        site.__class__ = cls
        return site

    def __init__(self, _site: MjsSite, name: str, index_x: int, index_y: int):
        if not hasattr(self, '_is_pheromone_field_cell_spec_initialized'):
            self._is_pheromone_field_cell_spec_initialized = True
            self.index_x = index_x
            self.index_y = index_y

    def get_cell(self, model: mujoco.MjModel) -> PheromoneFieldCell:
        site = model.site(self.name)
        return PheromoneFieldCell(site, self.index_x, self.index_y)


def add_pheromone_cell(
        spec: mujoco.MjSpec,
        index_x: int,
        index_y: int,
        size: float,
        pos: tuple[float, float, float],
        alpha: float,
) -> PheromoneFieldCellSpec:
    name = f"pheromone_cell_{index_x}_{index_y}"
    site_spec = add_site(
        spec.worldbody,
        name=name,
        pos=pos,
        size=[size, size, 1],
        rgba=(1.0, 1.0, 1.0, alpha),
        type_=mujoco.mjtGeom.mjGEOM_PLANE
    )
    return PheromoneFieldCellSpec(site_spec, name, index_x, index_y)
