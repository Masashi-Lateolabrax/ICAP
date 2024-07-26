import os.path

from scheme.pheromone_property_analysis.evaporation_vs_liquid import main

# # Task
# Pheromone Property Analysis
#   Evaporation vs. Liquid
#
# # Objective
# Analysing the maximum evaporation speed related to the amount of secreted pheromone.
#
# # Features
#   - EVAPORATION: 20.0
#   - DECREASE: 0.1
#   - DIFFUSION: 35.0
#   - SATURATION_VAPOR: 10.0
#   - LIQUID: [ 10,100,200,400,600,800,1000,10000,100000 ]
#   - TIMESTEP: 0.03
#   - EPISODE: 180
#   - SECRETING_PERIOD: 60
#
# # Difference
#   - SECRETING_PERIOD
#
# # Date
# 2024-07-26 18:35

if __name__ == '__main__':
    main(os.path.dirname(__file__))
