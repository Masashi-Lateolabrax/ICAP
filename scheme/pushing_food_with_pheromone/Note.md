# Experiment Note

## Task

Transportation with Pheromone

## Objective

To observe the behavior of the pheromone.

## Features

### Task

| Setting        | Value |
|----------------|-------|
| Episode Length | 30    |

### Robot Characteristic

| Settings       | Value                  |
|----------------|------------------------|
| Pheromone Tank | Disable                |
| RNN            | Enable                 |
| Dimension      | 244                    |
| Noise Layer    | Input and Middle Layer |

### Pheromone

| Settings         | Value              |
|------------------|--------------------|
| SATURATION_VAPOR | 5.442554913756431  |
| EVAPORATION      | 9.390048012289446  |
| DIFFUSION        | 0.4907347580652921 |  
| DECREASE         | 5.136664772909677  |

### Evaluation

| Settings        | Value |
|-----------------|-------|
| FOOD_RANGE      | 2.3   |
| FOOD_NEST_GAIN  | 50.0  |
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

- Enable the Noise Layer in the Middle layer.

## Date

2024-09-28 21:03 (JST)
