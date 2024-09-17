# Experiment Note

## Task

Transportation with Pheromone

## Objective

To achieve the robust training.

## Features

### Robot

| Setting   | Value  |
|-----------|--------|
| RNN       | Enable |
| Dimension | 684    |

### Task

| Setting        | Value |
|----------------|-------|
| Episode Length | 60    |

### Robot Characteristic

| Settings       | Value   |
|----------------|---------|
| Pheromone Tank | Disable |
| RNN            | Enable  |
| Dimension      | 1324    |
| Noise Layer    | Enable  |

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

- Shrink the robot brain size
- Decrease the population
- Increase the episode length

## Date

2024-09-17 23:33 (JST)
