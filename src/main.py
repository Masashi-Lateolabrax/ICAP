import os.path
import time

from scheme.pheromone_property_analysis.evaporation import main

# # Task
# Pheromone Property Analysis
#   Evaporation
#
# # Objective
# Analysing the effect of the evaporation coefficient effect on evaporation speed.
#
# # Features
#   - EVAPORATION: [i for i in range(5,151,5)]
#   - DECREASE: 0.1
#   - DIFFUSION: 35.0
#   - SATURATION_VAPOR: 10.0
#   - LIQUID: 1000000000
#   - TIMESTEP: 0.03
#   - EPISODE: 120
#
# # Changes
#   - Units of Evaporation Speed [/step -> /seconds]
#
# # Date
# 2024-07-30 04:28

if __name__ == '__main__':
    main(time.strftime("%Y%m%d_%H%M%S"), os.path.dirname(__file__))
