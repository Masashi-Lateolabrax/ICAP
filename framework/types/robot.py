from typing import Self

import jax
import jax.numpy as jnp
from flax.struct import field, dataclass as jax_dataclass

import mujoco
from mujoco import mjx


class RobotSpec:
    def __init__(
            self,
            body: mujoco.MjsBody,
            center_site: mujoco.MjsSite,
            front_site: mujoco.MjsSite,
            free_joint: mujoco.MjsJoint,
            x_act: mujoco.MjsActuator,
            y_act: mujoco.MjsActuator,
            z_act: mujoco.MjsActuator,
            r_act: mujoco.MjsActuator
    ):
        self.body = body
        self.center_site = center_site
        self.front_site = front_site
        self.free_joint = free_joint
        self.x_act = x_act
        self.y_act = y_act
        self.z_act = z_act
        self.r_act = r_act


@jax_dataclass
class RobotIDs:
    body_id: int = field(pytree_node=False)
    center_site_id: jnp.ndarray
    front_site_id: jnp.ndarray
    free_joint_id: jnp.ndarray
    x_actuator_id: jnp.ndarray
    y_actuator_id: jnp.ndarray
    z_actuator_id: jnp.ndarray
    r_actuator_id: jnp.ndarray


@jax_dataclass
class BatchedRobotIDs:
    body_ids: tuple[int, ...] = field(pytree_node=False)
    center_site_ids: jnp.ndarray
    front_site_ids: jnp.ndarray
    free_joint_ids: jnp.ndarray
    x_actuator_ids: jnp.ndarray
    y_actuator_ids: jnp.ndarray
    z_actuator_ids: jnp.ndarray
    r_actuator_ids: jnp.ndarray

    @classmethod
    def from_specs(cls, model: mujoco.MjModel | mjx.Model, specs: list[RobotSpec]) -> 'BatchedRobotIDs':
        body_ids = tuple([mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body.name) for spec in specs])
        center_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.center_site.name) for spec in specs]
        front_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.front_site.name) for spec in specs]
        free_joint_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.free_joint.name) for spec in specs]
        x_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.x_act.name) for spec in specs]
        y_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.y_act.name) for spec in specs]
        z_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.z_act.name) for spec in specs]
        r_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.r_act.name) for spec in specs]

        return cls(
            body_ids=body_ids,
            center_site_ids=jnp.array(center_site_ids, dtype=jnp.int32),
            front_site_ids=jnp.array(front_site_ids, dtype=jnp.int32),
            free_joint_ids=jnp.array(free_joint_ids, dtype=jnp.int32),
            x_actuator_ids=jnp.array(x_actuator_ids, dtype=jnp.int32),
            y_actuator_ids=jnp.array(y_actuator_ids, dtype=jnp.int32),
            z_actuator_ids=jnp.array(z_actuator_ids, dtype=jnp.int32),
            r_actuator_ids=jnp.array(r_actuator_ids, dtype=jnp.int32)
        )

    def __getitem__(self, index: int):
        return RobotIDs(
            body_id=self.body_ids[index],
            center_site_id=self.center_site_ids[index],
            front_site_id=self.front_site_ids[index],
            free_joint_id=self.free_joint_ids[index],
            x_actuator_id=self.x_actuator_ids[index],
            y_actuator_id=self.y_actuator_ids[index],
            z_actuator_id=self.z_actuator_ids[index],
            r_actuator_id=self.r_actuator_ids[index]
        )


@jax_dataclass
class RobotOutputs:
    left_wheel: jax.Array
    right_wheel: jax.Array
    pheromone: jax.Array

    @property
    def wheels(self) -> jax.Array:
        return jnp.stack([self.left_wheel, self.right_wheel], axis=1)

    @staticmethod
    def zeros(num_robots: int) -> "RobotOutputs":
        return RobotOutputs(
            left_wheel=jnp.zeros((num_robots,), dtype=jnp.float32),
            right_wheel=jnp.zeros((num_robots,), dtype=jnp.float32),
            pheromone=jnp.zeros((num_robots,), dtype=jnp.float32),
        )

    def fill(self, value: float) -> "RobotOutputs":
        return RobotOutputs(
            left_wheel=jnp.full(self.left_wheel.shape, value, dtype=jnp.float32),
            right_wheel=jnp.full(self.right_wheel.shape, value, dtype=jnp.float32),
            pheromone=jnp.full(self.pheromone.shape, value, dtype=jnp.float32),
        )

    def update(self, left_wheel=None, right_wheel=None, pheromone=None) -> Self:
        kwargs = {
            "left_wheel": left_wheel,
            "right_wheel": right_wheel,
            "pheromone": pheromone,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not kwargs:
            return self
        return self.replace(**kwargs)


@jax_dataclass
class RobotInputs:
    ray: jax.Array
    pheromone: jax.Array

    def as_matrix(self) -> jax.Array:
        return jnp.concatenate([self.ray, self.pheromone[:, None]], axis=1)

    @staticmethod
    def zeros(num_robots: int, num_ray: int) -> "RobotInputs":
        return RobotInputs(
            ray=jnp.zeros((num_robots, num_ray), dtype=jnp.float32),
            pheromone=jnp.zeros((num_robots,), dtype=jnp.float32),
        )

    def fill(self, value: float) -> Self:
        return RobotInputs(
            ray=jnp.full(self.ray.shape, value, dtype=jnp.float32),
            pheromone=jnp.full(self.pheromone.shape, value, dtype=jnp.float32),
        )

    def update(self, ray: jax.Array = None, pheromone: jax.Array = None) -> Self:
        kwargs = {
            "ray": ray,
            "pheromone": pheromone,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not kwargs:
            return self
        return self.replace(**kwargs)


@jax_dataclass
class BatchedRobots:
    ids: BatchedRobotIDs

    positions: jax.Array
    xdirections: jax.Array

    two_wheel_differential_move_matrix_T: jax.Array

    @classmethod
    def new(
            cls,
            data: mujoco.MjData | mjx.Data,
            batched_ids: BatchedRobotIDs,
            d: float,
            velocity: float
    ) -> 'BatchedRobots':
        this = cls(
            ids=batched_ids,
            positions=jnp.zeros((0, 3)),
            xdirections=jnp.zeros((0, 2)),
            two_wheel_differential_move_matrix_T=jnp.array([
                [velocity * 0.5, velocity * 0.5],
                [- velocity / d, velocity / d]
            ]).T
        )
        this = this.update(data)
        return this

    @property
    def num_robots(self) -> int:
        return self.positions.shape[0]

    @staticmethod
    @jax.jit
    def _update(data: mujoco.MjData | mjx.Data, ids: BatchedRobotIDs) -> tuple[jax.Array, jax.Array]:
        center_site_positions = data.site_xpos[ids.center_site_ids, :]
        front_site_positions = data.site_xpos[ids.front_site_ids, :2]
        sub = front_site_positions - center_site_positions[:, :2]
        n = jnp.linalg.norm(sub, axis=1, keepdims=True) + 1e-6
        xdirections = sub / n
        return center_site_positions, xdirections

    def update(self, data: mujoco.MjData | mjx.Data) -> 'BatchedRobots':
        positions, xdirections = self._update(data, self.ids)
        return self.replace(
            positions=positions,
            xdirections=xdirections
        )

    @jax.jit
    def _set_ctrl_mjx(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        power_and_torque = ctrl @ self.two_wheel_differential_move_matrix_T
        move = self.xdirections * power_and_torque[:, 0:1]
        torque = power_and_torque[:, 1]

        new_ctrl = data.ctrl.at[self.ids.x_actuator_ids].set(move[:, 0])
        new_ctrl = new_ctrl.at[self.ids.y_actuator_ids].set(move[:, 1])
        new_ctrl = new_ctrl.at[self.ids.r_actuator_ids].set(torque)
        return data.replace(ctrl=new_ctrl)

    def set_ctrl(self, data: mjx.Data, ctrl: jax.Array) -> mujoco.MjData | mjx.Data:
        return self._set_ctrl_mjx(data, ctrl)
