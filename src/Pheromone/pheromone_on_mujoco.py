from pheromone import Pheromone
from src.WrappedMujoco.deco_geom import DecoGeom


class PheromoneOnMujoco:
    def __init__(self, pheromone: Pheromone, width: int, height: int, x_pos: float, y_pos: float):
        self._pheromone = pheromone
        self.width = width
        self.height = height
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.tiles: list[list[DecoGeom | None]] = [[None] * height for _ in range(width)]

    def size(self) -> (int, int):
        size = self._pheromone.gas.size()
        return size[0], size[1]

    def _map_mujoco_pos_to_pheromone_index(self, x: float, y: float):
        pheromone_width, pheromone_height = self.size()
        dw = self.width / pheromone_width
        dh = self.height / pheromone_height
        nx = (x - (self.x_pos - pheromone_width * 0.5)) / dw
        ny = ((self.y_pos - pheromone_height * 0.5) - y) / dh
        return nx, ny

    def get_liquid_value(self, x: float, y: float) -> float:
        nx, ny = self._map_mujoco_pos_to_pheromone_index(x, y)
        return self._pheromone.get_liquid_value(nx, ny)

    def get_gas_value(self, x: float, y: float) -> float:
        nx, ny = self._map_mujoco_pos_to_pheromone_index(x, y)
        return self._pheromone.get_gas_value(nx, ny)

    def add_liquid(self, x: float, y: float, value: float):
        nx, ny = self._map_mujoco_pos_to_pheromone_index(x, y)
        self._pheromone.add_liquid(nx, ny, value)

    def step(self, dt: float):
        self._pheromone.step(dt)

    def set_tile(self, x: int, y: int, tile: DecoGeom):
        self.tiles[x][y] = tile

    def update_tile(self, func=lambda x: x):
        import colorsys
        for iy in range(self.height):
            for ix in range(self.width):
                gas = self._pheromone.get_raw_gas_value(ix, iy)
                v = func(gas)
                color = colorsys.hsv_to_rgb(0.66 * (1.0 - v), 1.0, 1.0)
                self.tiles[ix][iy].set_rgba((color[0], color[1], color[2], 0.7))
