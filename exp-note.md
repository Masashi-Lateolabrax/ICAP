# Memo

## 2025/07/19 15:52 8228d5a1

In this experiment, we employed Learning Rate Adaptation CMA-ES [Nomura et al., 2023].

The reason for this choice was the expectation that it could address the issue of parameters growing explosively during
training.

However, with this method, the loss value increased as the generations progressed, and from around the 100th generation,
it stabilized at approximately -550. Since the loss was about -900 in the first few generations, this indicates that the
evaluation worsened over time.

By the 500th generation, some of the parameters had become extremely large, reaching values close to -8000. Furthermore,
the parameter that reached the maximum value was not a single one; rather, several parameters alternated in taking large
values over the course of the optimization.

## 2025/07/18 11:48 a64cc661

We find a bug in the analysis code that overwrites incorrectly the timer variable in SimulatorForDebugging class.

As a result of this experiment, the robots moved only forward, and did not exhibit any foraging behavior.

We consider the small sigma value to be the cause of this issue. Therefore, we should search for appropriate sigma
values.

## 2025/07/16 11:05

We add six robots, so the number of robots is now nine.

We will try to train them as swarm robots.
