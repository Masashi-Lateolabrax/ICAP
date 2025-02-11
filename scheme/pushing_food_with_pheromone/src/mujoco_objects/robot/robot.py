from scheme.pushing_food_with_pheromone.lib.parts import OmniSensor, BrainJudgement, BrainInterface

from .data import Data
from .actuator import Actuator


class Robot:
    def __init__(
            self,
            brain: BrainInterface,
            body_,
            data: Data,
            actuator: Actuator,
            sensor: OmniSensor,
    ):
        self.brain = brain
        self.body = body_
        self._data = data
        self._actuator = actuator
        self._sensor = sensor

    def update(self):
        self._data.update()

    @property
    def position(self):
        return self._data.pos

    @property
    def direction(self):
        return self._data.direction

    def action(self):
        match self.brain.think():
            case BrainJudgement.STOP:
                self._actuator.stop()
            case BrainJudgement.FORWARD:
                self._actuator.forward()
            case BrainJudgement.BACK:
                self._actuator.back()
            case BrainJudgement.TURN_RIGHT:
                self._actuator.turn_right()
            case BrainJudgement.TURN_LEFT:
                self._actuator.turn_left()
            case BrainJudgement.SECRETION:
                self._actuator.secretion()
            case _:  # pragma: no cover
                raise ValueError("Invalid judge")
