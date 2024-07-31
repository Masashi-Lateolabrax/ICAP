import os.path
import time

from scheme.pheromone_property_analysis.sv_vs_ev_speed_curve import main

# # Task
# Pheromone Property Analysis
#   Evaporation Speed Curve
#
# # Objective
# Analysing the effect of the evaporation coefficient effect on evaporation speed.
#
# # Features
#   - EVAPORATION: [i for i in range(5,151,5)]
#   - DECREASE: 0.1
#   - DIFFUSION: 35.0
#   - SATURATION_VAPOR: [i for i in range(1,10)]
#   - LIQUID: 1000000000
#   - TIMESTEP: 0.03
#   - EPISODE: 120
#
# # Changes
#   - Nothing
#
# # Date
# 2024-07-31 21:49

if __name__ == '__main__':
    main(time.strftime("%Y%m%d_%H%M%S"), os.path.dirname(__file__))
