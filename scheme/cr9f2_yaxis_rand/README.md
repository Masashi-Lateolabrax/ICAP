# cr9f2 y axis rand

## Background

Our objective is to train a swarm of robots to perform food transport tasks. In this study, we addressed a relatively
simple problem called the `cr9f2-y-axis` task.

However, the robots leaned towards sequential behavior, effectively overfitting to the task. To mitigate this, we
introduced a randomization element and defined the new `cr9f2-y-axis-rand` task. Details of this variation are provided
in the Environment section.

## Environment

- **Robots**: Nine continuous-type robots are used.
- **Initial Setup**:
    - Positions: Robots start in the nest at the center of the simulation area.
    - Directions: Initially randomized, with the range increasing generation by generation.
- **Food Items**: Located at `(0, 6)` and `(0, -6)`.

## Optimization

We employ the CMA-ES algorithm for robot training.