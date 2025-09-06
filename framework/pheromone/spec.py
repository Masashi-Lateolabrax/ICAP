import mujoco

from ..mkenv import add_site
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
) -> PheromoneFieldCellSpec:
    name = f"pheromone_cell_{index_x}_{index_y}"
    site_spec = add_site(
        spec.worldbody,
        name=name,
        pos=pos,
        size=[size, size, 1e-6],
        rgba=(1.0, 1.0, 1.0, 1.0),
        type_=mujoco.mjtGeom.mjGEOM_BOX
    )
    return PheromoneFieldCellSpec(site_spec, index_x, index_y)


def add_pheromone_cells_to_mjspec(
        spec: mujoco.MjSpec,
        width_num: int,
        height_num: int,
        cell_size: float,
) -> list[PheromoneFieldCellSpec]:
    from ..pheromone import add_pheromone_cell

    sites = []
    for x in range(width_num):
        for y in range(height_num):
            pos_x = cell_size * (x - (width_num - 1) * 0.5)
            pos_y = cell_size * (-y + (height_num - 1) * 0.5)

            sites.append(
                add_pheromone_cell(
                    spec,
                    index_x=x,
                    index_y=y,
                    size=cell_size * 0.5,
                    pos=(pos_x, pos_y, 0),
                )
            )
    return sites
