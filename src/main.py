import os.path

from scheme.pheromone_property_analysis.exploration_of_parameters import main

# # Task
# Pheromone Property Analysis
#   Exploration of parameters
#
# # Objective
# Analysing the effect of the evaporation coefficient effect on evaporation speed.
#
# # Features
#   - EPISODE_LENGTH: 15
#   - TIMESTEP: 0.001
#
#   - CELL_SIZE_FOR_CALCULATION: 1
#   - CELL_SIZE_FOR_MUJOCO: 0.2
#
#   - GENERATION: 1000
#   - POPULATION: 10
#   - NUM_ELITE: 5
#   - SIGMA: 0.3
#
#   - STABLE_STATE_TIME: 5
#   - EVAPORATION_SPEED: 0.01
#   - RELATIVE_STABLE_GAS_VOLUME: 0.4
#   - DECREASED_STATE_TIME: 10
#   - FIELD_SIZE: 0.350
#
#   - EPS: 0.001
#   - STABILITY_THRESHOLD: 0.9999999
#
# # Changes
#   - Nothing
#
# # Date
# 2024-08-04 06:18

if __name__ == '__main__':
    main(os.path.dirname(__file__))
