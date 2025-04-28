from .simulator.objects.robot import BrainInterface


class Parameters:
    """
    A class to represent the parameters that will be optimized.

    Attributes:
    ----------
    brain : BrainInterface
        An instance of BrainInterface representing the robot's brain.
    """

    def __init__(self, brain: BrainInterface):
        self.brain = brain
