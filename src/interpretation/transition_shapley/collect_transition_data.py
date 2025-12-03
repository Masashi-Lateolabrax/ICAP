from src.config import Simulator
from src.interpretation.transition_shapley.data_structures import SimulatorState
from src.interpretation.transition_shapley import simulation


class TransitionCollector:
    def __init__(self, settings, individual, kmeans, scaler):
        self.simulator = Simulator(settings, individual, render=False)
        self.simulator_baseline = Simulator(settings, individual, render=False)
        self.kmeans = kmeans
        self.scaler = scaler

    def step(self):
        """Execute step and collect transition data if transitions occur.

        Returns:
            Tuple of (actual_next_input, baseline_next_input, transition_mask) where:
            - actual_next_input: (n_robots, 9) array of next inputs with actual pheromone
            - baseline_next_input: (n_robots, 9) array of next inputs with pheromone=0
            - transition_mask: (n_robots,) bool array indicating which robots transitioned

            Or None if no transitions occurred.
        """
        # Backup current state
        backup = SimulatorState(self.simulator)

        # Execute actual step and check for transitions
        transitions = simulation.step(self.simulator, self.kmeans, self.scaler)  # (n_robots,)

        if transitions.any():
            # At least one transition occurred! Get actual next input
            actual_next_input = self.simulator.input_ndarray.copy()  # (n_robots, 9)

            # Restore state to baseline simulator and execute baseline step
            backup.restore(self.simulator_baseline)
            baseline_next_input = simulation.step_baseline(self.simulator_baseline)  # (n_robots, 9)

            return actual_next_input, baseline_next_input, transitions

        return None