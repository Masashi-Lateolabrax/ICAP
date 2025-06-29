import abc


class SimulatorBackend(metaclass=abc.ABCMeta):
    def step(self):
        raise NotImplementedError
