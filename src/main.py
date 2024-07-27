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
#   - EVAPORATION: [ 1.0,5.0,10.0,50.0,100.0,500.0,1000.0,5000.0,10000.0 ]
#   - DECREASE: 0.1
#   - DIFFUSION: 35.0
#   - SATURATION_VAPOR: 10.0
#   - LIQUID: 1000000000
#   - TIMESTEP: 0.03
#   - EPISODE: 120
#
# # Changes
#   - Expand the EVAPORATION
#   - To ensure a sufficient amount of LIQUID, increase LIQUID.
#
# # Date
# 2024-07-27 23:11

if __name__ == '__main__':
    main(os.path.dirname(__file__))
