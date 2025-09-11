from functools import partial

from mujoco import mjx
import mujoco

import jax
import jax.numpy as jnp
from flax.struct import field, dataclass as jax_dataclass


class FoodSpec:
    def __init__(
            self,
            body: mujoco.MjsBody,
            center_site: mujoco.MjsSite,
            free_joint: mujoco.MjsJoint,
            velocimeter: mujoco.MjsSensor,
            x_act: mujoco.MjsActuator,
            y_act: mujoco.MjsActuator,
            z_act: mujoco.MjsActuator,
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
    body_id: jnp.ndarray = field(pytree_node=False)
    center_site_id: jnp.ndarray = field(pytree_node=False)
    free_joint_id: jnp.ndarray = field(pytree_node=False)
    velocimeter_id: jnp.ndarray = field(pytree_node=False)
    x_act_id: jnp.ndarray = field(pytree_node=False)
    y_act_id: jnp.ndarray = field(pytree_node=False)
    z_act_id: jnp.ndarray = field(pytree_node=False)


@jax_dataclass
class BatchedFoodIDs:
    body_ids: jnp.ndarray = field(pytree_node=False)
    center_site_ids: jnp.ndarray = field(pytree_node=False)
    free_joint_ids: jnp.ndarray = field(pytree_node=False)
    velocimeter_ids: jnp.ndarray = field(pytree_node=False)
    x_act_ids: jnp.ndarray = field(pytree_node=False)
    y_act_ids: jnp.ndarray = field(pytree_node=False)
    z_act_ids: jnp.ndarray = field(pytree_node=False)

    free_joint_qpos_adr: tuple[int, ...] = field(pytree_node=False)

    @classmethod
    def from_specs(cls, model: mjx.Model, specs: list[FoodSpec]) -> 'BatchedFoodIDs':
        body_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body.name) for spec in specs]
        center_site_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.center_site.name) for spec in specs]
        free_joint_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.free_joint.name) for spec in specs]
        velocimeter_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, spec.velocimeter.name) for spec in specs]
        x_act_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.x_act.name) for spec in specs]
        y_act_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.y_act.name) for spec in specs]
        z_act_ids = [mjx.name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.z_act.name) for spec in specs]

        free_joint_qpos_adr = tuple(int(model.jnt_qposadr[i]) for i in free_joint_ids)

        return cls(
            body_ids=jnp.array(body_ids, dtype=jnp.int32),
            center_site_ids=jnp.array(center_site_ids, dtype=jnp.int32),
            free_joint_ids=jnp.array(free_joint_ids, dtype=jnp.int32),
            velocimeter_ids=jnp.array(velocimeter_ids, dtype=jnp.int32),
            x_act_ids=jnp.array(x_act_ids, dtype=jnp.int32),
            y_act_ids=jnp.array(y_act_ids, dtype=jnp.int32),
            z_act_ids=jnp.array(z_act_ids, dtype=jnp.int32),
            free_joint_qpos_adr=free_joint_qpos_adr
        )

    def __getitem__(self, index: int) -> FoodIDs:
        return FoodIDs(
            body_id=self.body_ids[index],
            center_site_id=self.center_site_ids[index],
            free_joint_id=self.free_joint_ids[index],
            velocimeter_id=self.velocimeter_ids[index],
            x_act_id=self.x_act_ids[index],
            y_act_id=self.y_act_ids[index],
            z_act_id=self.z_act_ids[index],
        )


@jax_dataclass
class BatchedFood:
    num_food_items: int = field(pytree_node=False)
    ids: BatchedFoodIDs = field(pytree_node=False)
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
            num_food_items=batched_food_ids.body_ids.shape[0],
            ids=batched_food_ids,
            xmat=jnp.zeros((0, 3, 3), dtype=jnp.float32),
            positions=jnp.zeros((0, 3), dtype=jnp.float32),
            dummy_positions=jnp.zeros((0, 3), dtype=jnp.float32)
        )
        this = this.update(data)
        return this

    def update(self, data: mjx.Data) -> "BatchedFood":
        return self.replace(
            xmat=data.site_xmat[self.ids.center_site_ids],
            positions=data.site_xpos[self.ids.center_site_ids, :3]
        )

    @partial(jax.jit, static_argnames=["idx"], inline=True)
    def set_pos(self, data: mjx.Data, idx: int, pos: jax.Array) -> mjx.Data:
        qpos_addr = self.ids.free_joint_qpos_adr[idx]
        return data.replace(
            qpos=data.qpos.at[qpos_addr:qpos_addr + 3].set(pos[:3])
        )

    @partial(jax.jit, static_argnames=["idx"], inline=True)
    def set_force(self, data: mjx.Data, idx: int, force: jax.Array) -> mjx.Data:
        x_act_id = self.ids.x_act_ids[idx]
        y_act_id = self.ids.y_act_ids[idx]
        z_act_id = self.ids.z_act_ids[idx]

        new_ctrl = data.ctrl.at[x_act_id].set(force[0])
        new_ctrl = new_ctrl.at[y_act_id].set(force[1])
        new_ctrl = new_ctrl.at[z_act_id].set(force[2])

        return data.replace(ctrl=new_ctrl)
