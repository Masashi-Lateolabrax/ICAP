# Study Note

## Theme

Training robots to learn self-organized behavior using pheromones.

## Motivation

The study of self-organized behaviors in robots is crucial for developing autonomous systems that
can operate efficiently in complex environments.
By understanding the role pheromones play in behavior regulation,
we can design systems that mimic natural swarm intelligence,
leading to advancements in robotics, logistics, and communication.

## Contribution

- Developing a framework for training robots to achieve self-organized behavior.
- Evaluating the impact of pheromone mechanisms on robotic behavior.
- Providing insights into how environmental and systemic variables influence collective behavior in robotic systems.

## Objective

- Train robots to learn self-organised behavior.
- Evaluate how much pheromone contributes to the behavior.

## Road Map

1. Create simulation program.
2. Find settings for successful training.
3. Collect trained behaviors with various settings.
    - Test difference: sensor sensitivity, number of robots, number of food sources, environment size, loss function,
      hyperparameters of optimizer, robot's controller.
4. Define the index that represents how well robots are organized.
5. Analyze the relationship between settings and the organized indexes.
6. Convert the trained models into explainable models.
7. Analyze the effect of pheromones.

## Method

### Simulation

Use MuJoCo simulator.

Robots have two omni-sensors, pheromone secretion organ, two wheels, artificial neural network controller.

The omni-sensor is defined by the following formula.

$$
\mathrm{OmniSensor} = \frac{1}{N_\mathrm{O}} \sum^{}_{o \in \mathrm{Objects}}
\frac{1}{d_o + 1}
\begin{bmatrix}
\cos \theta_o \\
\sin \theta_o
\end{bmatrix} \,.
$$

Here,
*Objects* denote a set of sensing targets,
$N_\mathrm{O}$ denote the number of *Objects*,
$d_o$ denote a distance between the robot and $o$, and
$\theta_o$ denote a relative angle from the robot to $o$.

Note that I mean above *the robot* is observer.

Each robot has two omni-sensors because of allowing robots to observe the positions of other robots and food.
Namely, a robot has one with a set of robots and one with a set of food.

### Task

Robots work in the rectangle area, referred to as the field.

When create the field, Food are placed on the field randomly and
the nest is set at center of the field.

When food enters the nest, it is replaced randomly.

Robots are trained to transport food to the nest.

### Optimization Method

Use CMA-ES.

### Explainable Model

TODO

# Progress

1. Create simulation program.
    - [x] Create framework.
    - [x] Create the simplest environment example.
    - [ ] Implement a function to record the loss function output.
    - [ ] Examine CMA-ES applying to robots which take discrete actions.
2. Find settings for successful training.
3. Collect trained behaviors with various settings.
4. Define the index that represents how well robots are organized.
5. Analyze the relationship between settings and the organized indexes.
6. Convert the trained models into explainable models.
7. Analyze the effect of pheromones.

# For developers

## Git prefix

| Prefix | Definition                                                                                 |
|--------|--------------------------------------------------------------------------------------------|
| add    | Indicates that something has been added, such as files, directories, functions, and so on. |
| exp    | Indicates setup done for experiments.                                                      |
| mod    | Indicates changes with alternation of interfaces.                                          |
| doc    | Indicates that documentation has been added or modified.                                   |

## Branching strategy

| Name     | Description                                         | source       | merge to       |
|----------|-----------------------------------------------------|--------------|----------------|
| main     | Main branch                                         | -            | -              |
| develop  | Manage shared codes in scheme branches              | initial main | main, scheme/* |
| scheme/* | Manage codes of experiments for specific conditions | develop      | main           |

# Reference

Nothing yet.