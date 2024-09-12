# Experiment Note

## Task

Transportation with Pheromone

## Objective

To achieve the robust training.

## Features

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
- Randomize the initial robot directions in each generation for the optimization

## Changes

- Randomize the initial robot directions in each generation for the optimization
- Comment outed Pheromone Tank code

## Date

2024-09-13 08:03 (JST)
