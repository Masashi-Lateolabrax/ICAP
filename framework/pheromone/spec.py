import mujoco
from mujoco._specs import MjsSite

from ..environment import add_site
from .cell import PheromoneFieldCell


class PheromoneFieldCellSpec:
    def __init__(self, _site: MjsSite, index_x: int, index_y: int):
        self._site = _site
        self._index_x = index_x
        self._index_y = index_y

    @property
    def name(self) -> str:
        return self._site.name

    @property
    def index_x(self) -> int:
        return self._index_x

    @property
    def index_y(self) -> int:
        return self._index_y

    def get_cell(self, model: mujoco.MjModel) -> PheromoneFieldCell:
        site = model.site(self.name)
        return PheromoneFieldCell(site, self._index_x, self._index_y)


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
    return PheromoneFieldCellSpec(site_spec, index_x, index_y)
