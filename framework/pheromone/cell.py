from mujoco._structs import _MjModelSiteViews


class PheromoneFieldCell(_MjModelSiteViews):
    def __new__(cls, site: _MjModelSiteViews, index_x: int, index_y: int):
        site.__class__ = cls
        return site

    def __init__(self, _site: _MjModelSiteViews, index_x: int, index_y: int):
        if not hasattr(self, '_is_pheromone_field_cell_initialized'):
            self._is_pheromone_field_cell_initialized = True
            self.index_x = index_x
            self.index_y = index_y
            self.add_value = 0.0
