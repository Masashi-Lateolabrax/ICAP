import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

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


@dataclasses.dataclass
class RobotIDs:
    center_site_id: jnp.ndarray
    front_site_id: jnp.ndarray
    free_joint_id: jnp.ndarray
    x_actuator_id: jnp.ndarray
    y_actuator_id: jnp.ndarray
    z_actuator_id: jnp.ndarray
    r_actuator_id: jnp.ndarray


@dataclasses.dataclass
class BatchedRobotIDs:
    center_site_ids: jnp.ndarray
    front_site_ids: jnp.ndarray
    free_joint_ids: jnp.ndarray
    x_actuator_ids: jnp.ndarray
    y_actuator_ids: jnp.ndarray
    z_actuator_ids: jnp.ndarray
    r_actuator_ids: jnp.ndarray

    @classmethod
    def from_specs(cls, model: mujoco.MjModel | mjx.Model, specs: list[RobotSpec]) -> 'BatchedRobotIDs':
        center_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.center_site.name) for spec in specs]
        front_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.front_site.name) for spec in specs]
        free_joint_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.free_joint.name) for spec in specs]
        x_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.x_act.name) for spec in specs]
        y_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.y_act.name) for spec in specs]
        z_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.z_act.name) for spec in specs]
        r_actuator_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.r_act.name) for spec in specs]

        return cls(
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
            center_site_id=self.center_site_ids[index],
            front_site_id=self.front_site_ids[index],
            free_joint_id=self.free_joint_ids[index],
            x_actuator_id=self.x_actuator_ids[index],
            y_actuator_id=self.y_actuator_ids[index],
            z_actuator_id=self.z_actuator_ids[index],
            r_actuator_id=self.r_actuator_ids[index]
        )


class BatchedRobots:
    @staticmethod
    def __calc_power_and_torque(
            xdirections: jax.Array, ctrl: jax.Array,
            matrix_T: jax.Array
    ):
        power_and_torque = ctrl @ matrix_T
        move = xdirections * power_and_torque[:, 0]
        torque = power_and_torque[:, 1]
        return move, torque

    @staticmethod
    def __update_mjx_data_ctrl(
            data_: mjx.Data, xdirections: jax.Array, ctrl: jax.Array,
            matrix_T: jax.Array, x_actuator_ids: jax.Array, y_actuator_ids: jax.Array, r_actuator_ids: jax.Array,
    ):
        move, torque = BatchedRobots.__calc_power_and_torque(xdirections, ctrl, matrix_T)

        new_ctrl = data_.ctrl.at[x_actuator_ids].set(move[:, 0])
        new_ctrl = new_ctrl.at[y_actuator_ids].set(move[:, 1])
        new_ctrl = new_ctrl.at[r_actuator_ids].set(torque)

        return data_.replace(ctrl=new_ctrl)

    @staticmethod
    def __extract_state(
            data: mujoco.MjData | mjx.Data,
            center_site_ids: jax.Array,
            front_site_ids: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Extract positions and directions from mujoco data."""
        positions = data.site_xpos[center_site_ids, :2]
        front_pos = data.site_xpos[front_site_ids, :2]
        sub = front_pos - positions
        xdirections = sub / (jnp.linalg.norm(sub, axis=1, keepdims=True) + 1e-6)
        return positions, xdirections

    def __init__(
            self,
            data: mujoco.MjData | mjx.Data,
            batched_ids: BatchedRobotIDs,
            d: float,
            velocity: float,
    ):
        self.positions = data.site_xpos[batched_ids.center_site_ids, :2]

        front_site_pos = data.site_xpos[batched_ids.front_site_ids, :2]
        sub = front_site_pos - self.positions
        self.xdirections = sub / (jnp.linalg.norm(sub, axis=1, keepdims=True) + 1e-6)

        matrix_T = jnp.array([
            [velocity * 0.5, velocity * 0.5],
            [- velocity / d, velocity / d]
        ]).T
        self._calc_power_and_torque = jax.jit(partial(
            BatchedRobots.__calc_power_and_torque,
            matrix_T=matrix_T
        ))

        self.x_actuator_ids = batched_ids.x_actuator_ids.copy()
        self.y_actuator_ids = batched_ids.y_actuator_ids.copy()
        self.r_actuator_ids = batched_ids.r_actuator_ids.copy()

        self._update_mjx_data_ctrl = jax.jit(partial(
            BatchedRobots.__update_mjx_data_ctrl,
            matrix_T=matrix_T,
            x_actuator_ids=self.x_actuator_ids,
            y_actuator_ids=self.y_actuator_ids,
            r_actuator_ids=self.r_actuator_ids
        ))

        self._extract_state = partial(
            BatchedRobots.__extract_state,
            center_site_ids=batched_ids.center_site_ids,
            front_site_ids=batched_ids.front_site_ids
        )
        self._jit_extract_state = jax.jit(self._extract_state)

    @property
    def num_robots(self) -> int:
        return self.positions.shape[0]

    def update(self, data: mujoco.MjData | mjx.Data):
        if isinstance(data, mjx.Data):
            self.positions, self.xdirections = self._jit_extract_state(data)
        elif isinstance(data, mujoco.MjData):
            self.positions, self.xdirections = self._extract_state(data)

    def set_ctrl(self, data: mujoco.MjData | mjx.Data, ctrl: jax.Array):
        if isinstance(data, mjx.Data):
            data = self._update_mjx_data_ctrl(data, self.xdirections, ctrl)

        elif isinstance(data, mujoco.MjData):
            move, torque = self._calc_power_and_torque(self.xdirections, ctrl)
            data.ctrl[self.x_actuator_ids] = move[:, 0]
            data.ctrl[self.y_actuator_ids] = move[:, 1]
            data.ctrl[self.r_actuator_ids] = torque

        return data

    def tree_flatten(self):
        aux_data = {
            'x_actuator_ids': self.x_actuator_ids,
            'y_actuator_ids': self.y_actuator_ids,
            'r_actuator_ids': self.r_actuator_ids,
            'calc_power_and_torque': self._calc_power_and_torque,
            'update_mjx_data_ctrl': self._update_mjx_data_ctrl,
            'extract_state': self._extract_state,
            'jit_extract_state': self._jit_extract_state
        }
        return (self.positions, self.xdirections), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        positions, xdirections = children

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.x_actuator_ids = aux_data['x_actuator_ids']
        instance.y_actuator_ids = aux_data['y_actuator_ids']
        instance.r_actuator_ids = aux_data['r_actuator_ids']
        instance.positions = positions
        instance.xdirections = xdirections

        # Restore partial functions
        instance._calc_power_and_torque = aux_data['calc_power_and_torque']
        instance._update_mjx_data_ctrl = aux_data['update_mjx_data_ctrl']
        instance._extract_state = aux_data['extract_state']
        instance._jit_extract_state = aux_data['jit_extract_state']

        return instance


# Register BatchedRobots as JAX PyTree
jax.tree_util.register_pytree_node(
    BatchedRobots,
    BatchedRobots.tree_flatten,
    BatchedRobots.tree_unflatten
)
