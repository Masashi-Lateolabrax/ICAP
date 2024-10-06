# Experiment Note

## Task

Transportation with Pheromone

## Objective

To observe the behavior of the pheromone.

## Features

### Task

| Setting        | Value |
|----------------|-------|
| Episode Length | 60    |

### Robot Characteristic

| Settings       | Value                  |
|----------------|------------------------|
| Pheromone Tank | Disable                |
| RNN            | Enable                 |
| Dimension      | 244                    |
| Noise Layer    | Input and Middle Layer |

### Pheromone

| Settings         | Value |
|------------------|-------|
| SATURATION_VAPOR | 1     |
| EVAPORATION      | 1     |
| DIFFUSION        | 0.05  |  
| DECREASE         | 0.1   |

### Evaluation

| Settings        | Value |
|-----------------|-------|
| FOOD_RANGE      | 2.3   |
| FOOD_NEST_GAIN  | 5.0   |
| FOOD_ROBOT_GAIN | 2.756 |

- Use the distances between the nest and the food for the loss calculation

### Optimization

| Settings   | Value |
|------------|-------|
| Generation | 500   |
| Population | 50    |
| Elite      | 25    |

- Randomize the initial robot directions in each generation for the optimization

## Changes

- Change the pheromone characteristic.
- Fix the food positions.
- Reset the weights of the loss calculation.

## Date

2024-10-07 05:10 (JST)
