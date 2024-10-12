# Experiment Note

## Task

Transportation with Pheromone

## Objective

Collecting the evaluations of all individuals.

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

Potential

| Settings     | Value |
|--------------|-------|
| FOOD_GAIN    | 1     |
| FOOD_RANGE   | 6     |
| FOOD_RANGE_P | 2     |
| NEST_GAIN    | 7     |
| NEST_RANGE_P | 2     |

Distance

| Settings        | Value |
|-----------------|-------|
| FOOD_RANGE      | 2.3   |
| FOOD_NEST_GAIN  | 5.0   |
| FOOD_ROBOT_GAIN | 2.756 |

### Optimization

| Settings   | Value |
|------------|-------|
| Generation | 500   |
| Population | 50    |
| Elite      | 25    |

- Randomize the initial robot directions in each generation for the optimization
- Randomize the food positions

## Changes

- Remove Logger
- Increase GENERATION and POPULATION
- Use the latest evaluation formula (Distance)

## Date

2024-10-12 17:25 (JST)
