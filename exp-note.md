# Memo

## 2025/07/18 11:48 a64cc661

We find a bug in the analysis code that overwrites incorrectly the timer variable in SimulatorForDebugging class.

This attempt, as a result, causes the robot to move forward only.

As a result of this experiment, the robots moved only forward, and did not exhibit any foraging behavior.

We consider the small sigma value to be the cause of this issue. Therefore, we should search for appropriate sigma
values.

## 2025/07/16 11:05

We add six robots, so the number of robots is now nine.

We will try to train them as swarm robots.
