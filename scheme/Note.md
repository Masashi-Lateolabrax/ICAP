# Experiment Note

## Task

Transportation with Pheromone

## Objective

To achieve the robust training.

## Features

### Simulation

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

| Setting    | Value |
|------------|-------|
| Generation | 1000  |

- Randomize the initial robot directions in each generation for the optimization

## Changes

- Increase the number of generation
- Increase the episode length

## Date

2024-09-17 00:25 (JST)
