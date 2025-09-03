import numpy as np
from mujoco import mjx
import mujoco

import jax
import jax.numpy as jnp
from flax.struct import dataclass as jax_dataclass


class FoodSpec:
    def __init__(
            self,
            body: mujoco._specs.MjsBody,
            center_site: mujoco._specs.MjsSite,
            free_joint: mujoco._specs.MjsJoint,
            velocimeter: mujoco._specs.MjsSensor,
            x_act: mujoco._specs.MjsActuator,
            y_act: mujoco._specs.MjsActuator,
            z_act: mujoco._specs.MjsActuator,
    ):
        self.body = body
        self.center_site = center_site
        self.free_joint = free_joint
        self.velocimeter = velocimeter
        self.x_act = x_act
        self.y_act = y_act
        self.z_act = z_act


@jax_dataclass
class FoodIDs:
    body_ids: jnp.ndarray
    center_site_id: jnp.ndarray
    free_joint_id: jnp.ndarray
    velocimeter_id: jnp.ndarray
    x_act_id: jnp.ndarray
    y_act_id: jnp.ndarray
    z_act_id: jnp.ndarray


@jax_dataclass
class BatchedFoodIDs:
    body_ids: jnp.ndarray
    center_site_ids: jnp.ndarray
    free_joint_ids: jnp.ndarray
    velocimeter_ids: jnp.ndarray
    x_act_ids: jnp.ndarray
    y_act_ids: jnp.ndarray
    z_act_ids: jnp.ndarray

    free_joint_qpos_adr: jnp.ndarray

    @classmethod
    def from_specs(cls, model: mjx.Model, specs: list[FoodSpec]) -> 'BatchedFoodIDs':
        body_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body.name) for spec in specs]
        center_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.center_site.name) for spec in specs]
        free_joint_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.free_joint.name) for spec in specs]
        velocimeter_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, spec.velocimeter.name) for spec in specs]
        x_act_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.x_act.name) for spec in specs]
        y_act_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.y_act.name) for spec in specs]
        z_act_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.z_act.name) for spec in specs]

        free_joint_qpos_adr = [model.jnt_qposadr[i] for i in free_joint_ids]

        return cls(
            body_ids=jnp.array(body_ids, dtype=jnp.int32),
            center_site_ids=jnp.array(center_site_ids, dtype=jnp.int32),
            free_joint_ids=jnp.array(free_joint_ids, dtype=jnp.int32),
            velocimeter_ids=jnp.array(velocimeter_ids, dtype=jnp.int32),
            x_act_ids=jnp.array(x_act_ids, dtype=jnp.int32),
            y_act_ids=jnp.array(y_act_ids, dtype=jnp.int32),
            z_act_ids=jnp.array(z_act_ids, dtype=jnp.int32),
            free_joint_qpos_adr=jnp.array(free_joint_qpos_adr, dtype=jnp.int32),
        )

    def __getitem__(self, index: int) -> FoodIDs:
        return FoodIDs(
            body_ids=self.body_ids[index],
            center_site_id=self.center_site_ids[index],
            free_joint_id=self.free_joint_ids[index],
            velocimeter_id=self.velocimeter_ids[index],
            x_act_id=self.x_act_ids[index],
            y_act_id=self.y_act_ids[index],
            z_act_id=self.z_act_ids[index],
        )


@jax_dataclass
class BatchedFood:
    ids: BatchedFoodIDs
    xmat: jax.Array
    positions: jax.Array
    dummy_positions: jax.Array

    @classmethod
    def new(
            cls,
            data: mujoco.MjData | mjx.Data,
            batched_food_ids: BatchedFoodIDs,
    ):
        this = cls(
            ids=batched_food_ids,
            xmat=jnp.zeros((0, 3, 3), dtype=jnp.float32),
            positions=jnp.zeros((0, 3), dtype=jnp.float32),
            dummy_positions=jnp.zeros((0, 3), dtype=jnp.float32)
        )
        this = this.update(data)
        return this

    def update(self, data: mjx.Data) -> "BatchedFood":
        new_positions = data.site_xpos[self.ids.center_site_ids, :3]
        xmat = data.site_xmat[self.ids.center_site_ids]
        return self.replace(xmat=xmat, positions=new_positions)

    @staticmethod
    @jax.jit
    def _set_pos(data: mjx.Data, qpos_addr: jax.Array, pos: jax.Array) -> mjx.Data:
        new_qpos = jax.lax.dynamic_update_slice(data.qpos, pos[:3], (qpos_addr,))
        return data.replace(qpos=new_qpos)

    def set_pos(self, data: mjx.Data, idx: jax.Array, pos: jax.Array) -> mjx.Data:
        qpos_addr = self.ids.free_joint_qpos_adr[idx]
        return BatchedFood._set_pos(data, qpos_addr, pos)

    @staticmethod
    @jax.jit
    def _set_force(this: "BatchedFood", data: mjx.Data, idx: jax.Array, force: jax.Array) -> mjx.Data:
        x_act_id = this.ids.x_act_ids[idx]
        y_act_id = this.ids.y_act_ids[idx]
        z_act_id = this.ids.z_act_ids[idx]

        new_ctrl = data.ctrl.at[x_act_id].set(force[:, 0])
        new_ctrl = new_ctrl.at[y_act_id].set(force[:, 1])
        new_ctrl = new_ctrl.at[z_act_id].set(force[:, 2])

        return data.replace(ctrl=new_ctrl)

    def set_force(self, data: mjx.Data, idx: jax.Array, force: jax.Array) -> mjx.Data:
        return BatchedFood._set_force(self, data, idx, force)
