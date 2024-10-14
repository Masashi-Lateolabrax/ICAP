# Experiment Note

## Task

Transportation with Pheromone

## Objective

Collecting the evaluations of all individuals.

## Features

### Robot Characteristic

| Settings       | Value         |
|----------------|---------------|
| Pheromone Tank | Disable       |
| GRU            | Disable       |
| Dimension      | 134           |
| Noise Layer    | Input, Middle |

### Optimization

| Settings   | Value |
|------------|-------|
| Generation | 500   |
| Population | 50    |
| Elite      | 25    |

- Use 'Potential' evaluation.
- Randomize the initial robot directions in each generation for the optimization
- Randomize the food positions

### Evaluation

#### Potential

| Settings     | Value |
|--------------|-------|
| FOOD_GAIN    | 1     |
| FOOD_RANGE   | 6     |
| FOOD_RANGE_P | 2     |
| NEST_GAIN    | 7     |
| NEST_RANGE_P | 2     |

#### Distance

| Settings        | Value |
|-----------------|-------|
| FOOD_RANGE      | 2.3   |
| FOOD_NEST_GAIN  | 5.0   |
| FOOD_ROBOT_GAIN | 2.756 |

### Task

| Setting        | Value |
|----------------|-------|
| Episode Length | 60    |

### Pheromone

| Settings         | Value     |
|------------------|-----------|
| SATURATION_VAPOR | 1         |
| EVAPORATION      | 1         |
| DIFFUSION        | 0.05      |  
| DECREASE         | 0.1       |
| MAX_SECRETION    | 0.423 * 3 |

## Changes

- Disable RNN and use Liner
- Increase EPISODE from 45 to 60
- Decrease POPULATION from 100 to 50
- Decrease GENERATION from 1000 to 500

## Date

2024-10-14 04:38 (JST)
