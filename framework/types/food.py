import dataclasses
from functools import partial

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
            velocimeter: mujoco._specs.MjsSensor
    ):
        self.body = body
        self.center_site = center_site
        self.free_joint = free_joint
        self.velocimeter = velocimeter


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
        this = cls.__new__(cls)
        this = this.replace(
            ids=batched_food_ids,
            dummy_positions=jnp.zeros((0, 3), dtype=jnp.float32)
        )
        this = this.update(data)
        return this

    @property
    def positions_with_dummies(self):
        return jnp.vstack([self.positions, self.dummy_positions])

    def register_dummy_position(self, position: jax.Array) -> "BatchedFood":
        return self.replace(
            dummy_positions=jnp.vstack([self.dummy_positions, position]),
        )

    def update(self, data: mjx.Data | mujoco.MjData) -> "BatchedFood":
        new_positions = data.site_xpos[self.ids.center_site_ids, :3]
        return self.replace(positions=new_positions)

    @staticmethod
    def set_pos(data: mujoco.MjData | mjx.Data, body_id: jax.Array, pos: jax.Array):
        if isinstance(data, mjx.Data):
            new_xpos = data.xpos.at[body_id, :2].set(pos[:2])
            data = data.replace(xpos=new_xpos)

        elif isinstance(data, mujoco.MjData):
            data.xpos[body_id, :2] = np.array(pos[:2])

        return data

    # class BatchedFood:
    #     @staticmethod
    #     def __extract_from_data(
    #             data: mjx.Data,
    #             batched_food_ids: BatchedFoodIDs
    #     ):
    #         return data.site_xpos[batched_food_ids.center_site_ids, :2]
    #
    #     def __init__(
    #             self,
    #             data: mujoco.MjData | mjx.Data,
    #             batched_food_ids: BatchedFoodIDs,
    #     ):
    #         self.ids = batched_food_ids
    #         self.positions = data.site_xpos[batched_food_ids.center_site_ids, :2]
    #         self.dummy_positions = jnp.zeros((0, 2), dtype=jnp.float32)
    #
    #         self._extract_from_data = partial(
    #             BatchedFood.__extract_from_data,
    #             batched_food_ids=batched_food_ids
    #         )
    #         self._jit_extract_from_data = jax.jit(self._extract_from_data)
    #
    #     @property
    #     def positions_with_dummies(self):
    #         return jnp.vstack([self.positions, self.dummy_positions])
    #
    #     def register_dummy_position(self, position: jax.Array):
    #         self.dummy_positions = jnp.vstack([self.dummy_positions, position])
    #
    #     def update(self, data: mjx.Data | mujoco.MjData):
    #         if isinstance(data, mjx.Data):
    #             self.positions = self._jit_extract_from_data(data)
    #
    #         elif isinstance(data, mujoco.MjData):
    #             self.positions = self._extract_from_data(data)
    #
    #     @staticmethod
    #     def set_pos(data: mujoco.MjData | mjx.Data, body_id: jax.Array, pos: jax.Array):
    #         if isinstance(data, mjx.Data):
    #             new_xpos = data.xpos.at[body_id, :2].set(pos)
    #             data = data.replace(xpos=new_xpos)
    #
    #         elif isinstance(data, mujoco.MjData):
    #             data.xpos[body_id] = np.array(pos)
    #
    #         return data
    #
    #     def tree_flatten(self):
    #         leaves = (
    #             self.positions,
    #             self.dummy_positions
    #         )
    #         aux_data = {
    #             'ids': self.ids,
    #             'extract_from_data': self._extract_from_data,
    #             'jit_extract_from_data': self._jit_extract_from_data
    #         }
    #         return leaves, aux_data
    #
    #     @classmethod
    #     def tree_unflatten(cls, aux_data, children):
    #         positions, dummy_positions = children
    #
    #         instance = cls.__new__(cls)
    #         instance.ids = aux_data['ids']
    #         instance.positions = positions
    #         instance.dummy_positions = dummy_positions
    #         instance._extract_from_data = aux_data['extract_from_data']
    #         instance._jit_extract_from_data = aux_data['jit_extract_from_data']
    #
    #         return instance
    #
    #
    # # Register BatchedFood as JAX PyTree
    # jax.tree_util.register_pytree_node(
    #     BatchedFood,
    #     BatchedFood.tree_flatten,
    #     BatchedFood.tree_unflatten
    # )
