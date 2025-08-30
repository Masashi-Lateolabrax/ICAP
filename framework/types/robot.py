from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass as jax_dataclass

import mujoco
from mujoco import mjx


class RobotSpec:
    def __init__(
            self,
            body: mujoco._specs.MjsBody,
            center_site: mujoco._specs.MjsSite,
            front_site: mujoco._specs.MjsSite,
            free_joint: mujoco._specs.MjsJoint,
            x_act: mujoco._specs.MjsActuator,
            y_act: mujoco._specs.MjsActuator,
            z_act: mujoco._specs.MjsActuator,
            r_act: mujoco._specs.MjsActuator
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
    body_id: jnp.ndarray
    center_site_id: jnp.ndarray
    front_site_id: jnp.ndarray
    free_joint_id: jnp.ndarray
    x_actuator_id: jnp.ndarray
    y_actuator_id: jnp.ndarray
    z_actuator_id: jnp.ndarray
    r_actuator_id: jnp.ndarray


@jax_dataclass
class BatchedRobotIDs:
    body_ids: jnp.ndarray
    center_site_ids: jnp.ndarray
    front_site_ids: jnp.ndarray
    free_joint_ids: jnp.ndarray
    x_actuator_ids: jnp.ndarray
    y_actuator_ids: jnp.ndarray
    z_actuator_ids: jnp.ndarray
    r_actuator_ids: jnp.ndarray

    @classmethod
    def from_specs(cls, model: mujoco.MjModel | mjx.Model, specs: list[RobotSpec]) -> 'BatchedRobotIDs':
        body_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body.name) for spec in specs]
        center_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.center_site.name) for spec in specs]
        front_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.front_site.name) for spec in specs]
        free_joint_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.free_joint.name) for spec in specs]
        x_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.x_act.name) for spec in specs]
        y_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.y_act.name) for spec in specs]
        z_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.z_act.name) for spec in specs]
        r_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.r_act.name) for spec in specs]

        return cls(
            body_ids=jnp.array(body_ids, dtype=jnp.int32),
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
        this = cls.__new__(cls)
        this = this.replace(
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

    def set_ctrl(self, data: mujoco.MjData | mjx.Data, ctrl: jax.Array) -> mujoco.MjData | mjx.Data:
        if isinstance(data, mjx.Data):
            return self._set_ctrl_mjx(data, ctrl)

        elif isinstance(data, mujoco.MjData):
            power_and_torque = ctrl @ self.two_wheel_differential_move_matrix_T
            move = self.xdirections * power_and_torque[:, 0:1]
            torque = power_and_torque[:, 1]

            data.ctrl[self.ids.x_actuator_ids] = move[:, 0]
            data.ctrl[self.ids.y_actuator_ids] = move[:, 1]
            data.ctrl[self.ids.r_actuator_ids] = torque

            return data

        raise TypeError("data must be of type mujoco.MjData or mjx.Data")
