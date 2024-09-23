# Experiment Note

## Task

Transportation with Pheromone

## Objective

To achieve the robust training.

## Features

### Task

| Setting        | Value |
|----------------|-------|
| Episode Length | 60    |

### Robot Characteristic

| Settings       | Value   |
|----------------|---------|
| Pheromone Tank | Disable |
| RNN            | Enable  |
| Dimension      | 244     |
| Noise Layer    | Disable |

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

- Disable Noise layer

## Date

2024-09-20 14:07 (JST)
