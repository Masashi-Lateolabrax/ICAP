import os.path

from scheme.pheromone_property_analysis.evaporation import main

# # Task
# Pheromone Property Analysis
#   Evaporation
#
# # Objective
# Analysing the effect of the evaporation coefficient effect on evaporation speed.
#
# # Features
#   - EVAPORATION: [i for i in range(5,501,5)]
#   - DECREASE: 0.1
#   - DIFFUSION: 35.0
#   - SATURATION_VAPOR: 10.0
#   - LIQUID: 1000000000
#   - TIMESTEP: 0.03
#   - EPISODE: 120
#
# # Changes
#   - EVAPORATION has values from 5 to 500 in steps of 5
#
# # Date
# 2024-07-29 00:30

if __name__ == '__main__':
    main(os.path.dirname(__file__))
