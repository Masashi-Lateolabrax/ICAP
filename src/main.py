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
#   - LIQUID: [ 10,50,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 ]
#
# # Difference
#   - NONE
#
# # Date
# 2024-07-24 10:27

if __name__ == '__main__':
    main(os.path.dirname(__file__))
