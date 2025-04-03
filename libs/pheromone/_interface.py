import abc


class PheromoneCellInf(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_gas(self, value):
        raise NotImplemented

    @abc.abstractmethod
    def set_liquid(self, value):
        raise NotImplemented

    @abc.abstractmethod
    def get_gas(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_liquid(self):
        raise NotImplemented


class PheromoneFieldInf(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_field_size(self) -> tuple[int, int]:
        raise NotImplemented

    @abc.abstractmethod
    def get_sv(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_cell(self, xi: int, yi: int) -> PheromoneCellInf:
        raise NotImplemented

    @abc.abstractmethod
    def get_all_liquid(self):
        raise NotImplemented

    @abc.abstractmethod
    def set_liquid(self, xi: int, yi: int, value: float):
        raise NotImplemented

    @abc.abstractmethod
    def add_liquid(self, xi: float, yi: float, value: float):
        raise NotImplemented

    @abc.abstractmethod
    def get_all_gas(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_gas(self, xi: float, yi: float) -> float:
        raise NotImplemented

    @abc.abstractmethod
    def update(self, dt: float, iteration: int = 1):
        raise NotImplemented
