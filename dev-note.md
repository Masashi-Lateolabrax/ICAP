# Memo

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