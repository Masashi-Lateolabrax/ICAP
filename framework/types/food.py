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


@jax_dataclass
class BatchedFoodIDs:
    body_ids: jnp.ndarray
    center_site_ids: jnp.ndarray
    free_joint_ids: jnp.ndarray
    velocimeter_ids: jnp.ndarray

    @classmethod
    def from_specs(cls, model: mujoco.MjModel | mjx.Model, specs: list[FoodSpec]) -> 'BatchedFoodIDs':
        body_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body.name) for spec in specs]
        center_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.center_site.name) for spec in specs]
        free_joint_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.free_joint.name) for spec in specs]
        velocimeter_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, spec.velocimeter.name) for spec in specs]

        return cls(
            body_ids=jnp.array(body_ids, dtype=jnp.int32),
            center_site_ids=jnp.array(center_site_ids, dtype=jnp.int32),
            free_joint_ids=jnp.array(free_joint_ids, dtype=jnp.int32),
            velocimeter_ids=jnp.array(velocimeter_ids, dtype=jnp.int32)
        )

    def __getitem__(self, index: int) -> FoodIDs:
        return FoodIDs(
            body_ids=self.body_ids[index],
            center_site_id=self.center_site_ids[index],
            free_joint_id=self.free_joint_ids[index],
            velocimeter_id=self.velocimeter_ids[index]
        )


@jax_dataclass
class BatchedFood:
    ids: BatchedFoodIDs
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
            positions=jnp.zeros((0, 3), dtype=jnp.float32),
            dummy_positions=jnp.zeros((0, 3), dtype=jnp.float32)
        )
        this = this.update(data)
        return this

    def update(self, data: mjx.Data | mujoco.MjData) -> "BatchedFood":
        new_positions = data.site_xpos[self.ids.center_site_ids, :3]
        return self.replace(positions=new_positions)

    @staticmethod
    @jax.jit
    def _set_pos(data: mjx.Data, body_id: jax.Array, pos: jax.Array) -> mjx.Data:
        new_xpos = data.xpos.at[body_id, :3].set(pos[:3])
        data = data.replace(xpos=new_xpos)
        return data

    def set_pos(self, data: mjx.Data, idx: int, pos: jax.Array) -> mjx.Data:
        body_id = self.ids.body_ids[idx]
        return BatchedFood._set_pos(data, body_id, pos)
