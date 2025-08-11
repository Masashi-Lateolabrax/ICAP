from mujoco._structs import _MjModelSiteViews


class PheromoneFieldCell:
    def __init__(self, site: _MjModelSiteViews, index_x: int, index_y: int):
        self._site = site
        self.index_x = index_x
        self.index_y = index_y
        self.add_value = 0.0

    def set_color(self, r: float, g: float, b: float, a: float):
        self._site.rgba = [r, g, b, a]
