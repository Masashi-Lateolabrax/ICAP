import os.path

from scheme.pheromone_property_analysis.evaporation import main

# # Task
# Pheromone Property Analysis
#   Evaporation
#
# # Objective
# Analysing the effect of the evaporation coefficient effect to evaporation speed.
#
# # Features
#   - EVAPORATION: [ 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0 ]
#   - DECREASE: 0.1
#   - DIFFUSION: 35.0
#   - SATURATION_VAPOR: 10.0
#   - LIQUID: 100000
#   - TIMESTEP: 0.03
#   - EPISODE: 120
#
# # Difference
#   - Nothing
#
# # Date
# 2024-07-27 09:49

if __name__ == '__main__':
    main(os.path.dirname(__file__))
