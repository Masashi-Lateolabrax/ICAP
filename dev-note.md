# Memo

## 2025/07/19 19:22 - Food Relocation Pattern Study

**Experimental Design**: Two food relocation mechanisms have been implemented to investigate the impact of food
relocation patterns on robotic collective behavior learning.

### Food Relocation Mechanism Classification

- **Pattern-type Relocation**: All individuals within the same generation experience identical food placement patterns.
  Patterns change randomly between generations.
- **Random-type Relocation**: Individuals within the same generation experience different food placement patterns. Food
  is randomly relocated whenever carried to the nest.

### Experimental History and Comparative Study

- **Previous Experiment** (commit c2fbdac5): Pattern-type relocation
- **Current Experiment**: Changed to random-type relocation
- **Key Changes**:
    - Food placement: Pattern-type → Random-type (introduction of environmental stochasticity)

**Objective**: Through comparative experiments between pattern-type and random-type relocations, verify the potential
impact of inter-individual randomness on collective behavior emergence.

### Implementation Status

**Completed:**
- [x] Food relocation mechanism implementation (both pattern-type and random-type)
- [x] Configuration switch to random-type relocation

**New Tasks:**
- [ ] None

## 2025/07/17 17:40

We have implemented several features for analysis and changed the settings to be
similar to my past study settings.

### Done

- [x] Implement plotting of other robot sensor inputs
- [x] Fix the initial robot distribution
- [x] Implement THINK_INTERVAL mechanism
- [x] Correct the regularization of Preprocessed Omni Sensor
- [x] Add a food item
- [x] Implement the analysis code for fitness graphs
- [x] Implement the analysis code for robot parameters
- [x] Change the settings:
    - Optimization.POPULATION: from 1000 to 100
    - Optimization.SIGMA: from 0.5 to 0.01
    - Robot.THINK_INTERVAL: 0.05
    - Simulation.TIME_LENGTH: from 60 to 45
    - Storage.TOP_N: from 1 to 5

### TODO

- [ ] Instantiate the setting class and distribute it to clients
- [ ] Implement Pattern Relocation Mechanism
- [ ] Implement Flexible Object Size Setting

## 2025/07/16 10:49

Currently, we can train two robots, so the next challenge is to train nine robots.

The analysis code supports recording training results and animating robot positions, food positions, and relative nest
directions. We will implement plotting of food sensor inputs.

### TODO

- [x] Implement training of nine robots
- [x] Implement plotting of food sensor inputs