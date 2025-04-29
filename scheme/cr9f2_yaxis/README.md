# cr9f2 y axis

## Background

Our objective is to train a swarm of robots to perform food transport tasks.

In this study, we focus on a relatively simple problem,
which we refer to as the `cr9f2-y-axis` task.

The details of the `cr9f2-y-axis` task are provided in the Environment section.

## Environment

There are nine robots and two food items.

The robots are of a continuous type.

The robots are initially placed in the nest, which is located at the center of
the simulation area.

The food items are positioned along the y-axis, excluding the nest area.

At the beginning of each generation, the initial direction of each robot and
the initial position of each food item are randomized.

## Optimization

We use CMA-ES for robot training.
