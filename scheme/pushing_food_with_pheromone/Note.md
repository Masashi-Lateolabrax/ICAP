# Experiment Note

## Task

Transportation with Pheromone

## Objective

To find the best optimization settings.

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

| Settings     | Value |
|--------------|-------|
| FOOD_GAIN    | 1     |
| FOOD_RANGE   | 6     |
| FOOD_RANGE_P | 2     |
| NEST_GAIN    | 7     |
| NEST_RANGE_P | 2     |

### Optimization

| Settings   | Value |
|------------|-------|
| Generation | 500   |
| Population | 50    |
| Elite      | 25    |

- Randomize the initial robot directions in each generation for the optimization

## Changes

- Use the potential evaluation function
- Maximizing the evaluations

## Date

2024-10-08 06:05 (JST)
