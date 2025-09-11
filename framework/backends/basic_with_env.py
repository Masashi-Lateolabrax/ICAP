from functools import partial
import os
from typing import Self

import mujoco
import mujoco.mjx as mjx

import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import field, dataclass as jax_dataclass

from ..prelude import *
from ..mkenv import (
    add_geom,
    setup_option, setup_visual, setup_textures, add_nest, add_wall,
    rand_robot_pos, rand_food_pos, add_mesh_in_asset, MeshContentType,
    add_food_object_with_mesh, add_robot_with_mesh,
)
from ..pheromone import PheromoneField
from .basic import BasicSimulator
from .utils import emit_rays


def generate_mjspec(
        settings: Settings
) -> tuple[
    mujoco.MjSpec,
    mujoco.MjsSite,
    list[RobotSpec],
    list[FoodSpec]
]:
    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    setup_visual(spec, settings)
    setup_textures(spec, settings)

    add_wall(spec, settings)

    add_geom(
        spec.worldbody,
        geom_type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(
            settings.Simulation.WORLD_WIDTH * 0.5,
            settings.Simulation.WORLD_HEIGHT * 0.5,
            1
        ),
        material="ground",
        rgba=GROUND_COLOR,
        condim=GROUND_COLLISION_CONDIM
    )

    nest_spec = add_nest(spec, settings)

    invalid_area: list[tuple[Position, float]] = []

    # Create food objects
    food_specs = []
    if settings.Food.NUM > 0:
        food_mesh = add_mesh_in_asset(
            spec,
            name="food_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "food-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        for i in range(settings.Food.NUM):
            if i < len(settings.Food.INITIAL_POSITION):
                position: Position = settings.Food.INITIAL_POSITION[i]
            else:
                position: Position = rand_food_pos(settings, invalid_area)
            invalid_area.append(
                (position, settings.Food.RADIUS)
            )
            food_specs.append(
                add_food_object_with_mesh(spec, settings, food_mesh, i, position)
            )

    # Create robots
    robot_specs = []
    if settings.Robot.NUM > 0:
        robot_mesh = add_mesh_in_asset(
            spec,
            name="robot_mesh",
            file=os.path.abspath(os.path.join(settings.Storage.ASSET_DIRECTORY, "robot-object.stl")),
            content_type=MeshContentType.STL,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        )

        for i in range(settings.Robot.NUM):
            if i < len(settings.Robot.INITIAL_POSITION):
                position: RobotLocation = settings.Robot.INITIAL_POSITION[i]
            else:
                position: RobotLocation = rand_robot_pos(settings, invalid_area)
            invalid_area.append(
                (position.position, settings.Robot.RADIUS)
            )
            robot_specs.append(
                add_robot_with_mesh(spec, settings, robot_mesh, i, position)
            )

    return spec, nest_spec, robot_specs, food_specs


@jax_dataclass
class Consts:
    WORLD_WIDTH: float
    WORLD_HEIGHT: float

    NEST_POSITION: jax.Array
    NEST_RADIUS: float

    FOOD_RADIUS: float

    NUM_RAYS: int = field(pytree_node=False)

    OFFSET_FOOD_AND_ROBOT: float
    SIGMA_FOOD_AND_ROBOT: float
    GAIN_FOOD_AND_ROBOT: float

    OFFSET_FOOD_AND_NEST: float
    SIGMA_FOOD_AND_NEST: float
    GAIN_FOOD_AND_NEST: float


@jax_dataclass
class BasicSimulatorWithEnv(SimPheromoneTrait, SimEvaluateTrait, SimRenderTrait):
    consts: Consts

    _parent_sim: BasicSimulator

    robots: BatchedRobots
    food_items: BatchedFood

    robot_inputs: RobotInputs
    robot_outputs: RobotOutputs
    _robot_pheromone_x_idx: jax.Array
    _robot_pheromone_y_idx: jax.Array

    loss: jax.Array

    rngs_for_relocating_food: jax.Array
    _loss_offset: jax.Array

    @property
    def NEST_POSITION(self) -> jax.Array:
        return self.consts.NEST_POSITION

    @property
    def data(self) -> mjx.Data:
        return self._parent_sim.data

    @property
    def pheromone(self) -> PheromoneField:
        return self._parent_sim.pheromone

    @property
    def loss_offset(self) -> jax.Array:
        return self._loss_offset

    def _update_parent(self, **kwargs: dict) -> Self:
        return self.replace(
            _parent_sim=self._parent_sim.update(**kwargs)
        )

    def update(
            self,
            data: mjx.Data = None,
            pheromone: PheromoneField = None,

            robots: BatchedRobots = None,
            food_items: BatchedFood = None,
            robot_inputs: RobotInputs = None,
            robot_outputs: RobotOutputs = None,
            loss: jax.Array = None,
            rngs_for_relocating_food: jax.Array = None,
            **kwargs
    ) -> Self:
        kwargs["data"] = data
        kwargs["pheromone"] = pheromone

        kwargs["robots"] = robots
        kwargs["food_items"] = food_items
        kwargs["robot_inputs"] = robot_inputs
        kwargs["robot_outputs"] = robot_outputs
        kwargs["loss"] = loss
        kwargs["rngs_for_relocating_food"] = rngs_for_relocating_food
        return self._update(**kwargs)

    @classmethod
    def new(cls, settings: Settings, rngs: jax.Array) -> tuple[mujoco.MjModel, Self]:
        mj_spec, nest_spec, robot_specs, food_specs = generate_mjspec(settings)
        mj_model, parent_sim = BasicSimulator.new(mj_spec, settings)

        batched_robot_id = BatchedRobotIDs.from_specs(mj_model, robot_specs)
        batched_food_id = BatchedFoodIDs.from_specs(mj_model, food_specs)

        distance_between_wheels = settings.Robot.DISTANCE_BETWEEN_WHEELS
        max_speed = settings.Robot.MAX_SPEED
        robots = BatchedRobots.new(parent_sim.data, batched_robot_id, distance_between_wheels, max_speed)

        food_items: BatchedFood = BatchedFood.new(parent_sim.data, batched_food_id)

        robot_inputs = RobotInputs.zeros(robots.num_robots, settings.Robot.NUM_RAYS)
        robot_outputs = RobotOutputs.zeros(robots.num_robots)

        return mj_model, cls(
            _parent_sim=parent_sim,

            robots=robots,
            food_items=food_items,

            robot_inputs=robot_inputs,
            robot_outputs=robot_outputs,
            _robot_pheromone_x_idx=jnp.zeros((robots.num_robots,), dtype=jnp.int32),
            _robot_pheromone_y_idx=jnp.zeros((robots.num_robots,), dtype=jnp.int32),

            loss=jnp.zeros((1,), dtype=jnp.float32),

            rngs_for_relocating_food=rngs,
            _loss_offset=jnp.zeros((1,), dtype=jnp.float32),

            consts=Consts(
                WORLD_WIDTH=settings.Simulation.WORLD_WIDTH,
                WORLD_HEIGHT=settings.Simulation.WORLD_HEIGHT,
                NEST_POSITION=settings.Nest.POSITION.as_array(),
                NEST_RADIUS=settings.Nest.RADIUS,
                FOOD_RADIUS=settings.Food.RADIUS,
                NUM_RAYS=settings.Robot.NUM_RAYS,
                OFFSET_FOOD_AND_ROBOT=settings.Loss.OFFSET_FOOD_AND_ROBOT,
                SIGMA_FOOD_AND_ROBOT=settings.Loss.SIGMA_FOOD_AND_ROBOT,
                GAIN_FOOD_AND_ROBOT=settings.Loss.GAIN_FOOD_AND_ROBOT,
                OFFSET_FOOD_AND_NEST=settings.Loss.OFFSET_FOOD_AND_NEST,
                SIGMA_FOOD_AND_NEST=settings.Loss.SIGMA_FOOD_AND_NEST,
                GAIN_FOOD_AND_NEST=settings.Loss.GAIN_FOOD_AND_NEST,
            )
        )

    @partial(jax.jit, inline=True)
    def _check_food_in_nest(self) -> jax.Array:
        diff = self.food_items.positions[:, :2] - self.consts.NEST_POSITION
        distance_squared = jnp.sum(diff * diff, axis=1)
        mask = distance_squared < (self.consts.NEST_RADIUS ** 2)
        return mask

    @partial(jax.jit, inline=True)
    def _generate_new_food_position(self, rngs: jax.Array) -> jax.Array:
        key, rngs = jax.random.split(rngs)
        random_xy = jax.random.uniform(
            key,
            shape=(2,),
            minval=self.consts.NEST_RADIUS,
            maxval=jnp.array([self.consts.WORLD_WIDTH, self.consts.WORLD_HEIGHT]) * 0.5 - self.consts.FOOD_RADIUS
        )

        key, rngs = jax.random.split(rngs)
        sign = 2 * jax.random.randint(key, shape=(3,), minval=0, maxval=2).astype(jnp.float32) - 1
        random_xy = sign.at[:2].multiply(random_xy)
        random_xy = random_xy.at[2].set(2.0)

        return random_xy

    @partial(jax.jit, inline=True)
    def _relocate_food_items(self) -> tuple[mjx.Data, jax.Array, jax.Array]:
        done_relocate = self._check_food_in_nest()
        not_relocate = jnp.logical_not(done_relocate)
        new_rngs = self.rngs_for_relocating_food
        data = self.data

        for idx in range(self.food_items.num_food_items):
            next_rngs, rngs = jax.random.split(new_rngs)
            relocated_position = self._generate_new_food_position(rngs)
            current_position = self.food_items.positions[idx]

            new_position = done_relocate[idx] * relocated_position + not_relocate[idx] * current_position
            new_rngs = done_relocate[idx] * next_rngs + not_relocate[idx] * new_rngs

            data = self.food_items.set_pos(data, idx, new_position)

        return data, new_rngs, done_relocate

    @staticmethod
    @partial(jax.jit, inline=True)
    def _calc_loss_between_food_and_robots(
            food_position: jax.Array,
            robot_positions: jax.Array,
            const: Consts
    ) -> jax.Array:
        subs = (robot_positions[:, :2] - food_position[None, :2])
        distance_squared = jnp.sum(subs * subs, axis=1) - const.OFFSET_FOOD_AND_ROBOT ** 2
        distance_squared = jnp.maximum(distance_squared, 0)
        return -jnp.sum(jnp.exp(-distance_squared / const.SIGMA_FOOD_AND_ROBOT)) * const.GAIN_FOOD_AND_ROBOT

    @staticmethod
    @partial(jax.jit, inline=True)
    def _calc_loss_between_food_and_nest(
            food_position: jax.Array,
            const: Consts
    ) -> jax.Array:
        diff = food_position[:2] - const.NEST_POSITION[:2]
        distance_squared = jnp.sum(diff * diff) - const.OFFSET_FOOD_AND_NEST ** 2
        distance_squared = jnp.maximum(distance_squared, 0)
        return -jnp.sum(jnp.exp(-distance_squared / const.SIGMA_FOOD_AND_NEST)) * const.GAIN_FOOD_AND_NEST

    def _action_step(self) -> Self:
        return self.update(
            data=self.robots.set_ctrl(self.data, self.robot_outputs.wheels),
            pheromone=self.pheromone.add_liquid(
                self._robot_pheromone_x_idx, self._robot_pheromone_y_idx, self.robot_outputs.pheromone
            )
        )

    def _update_step(self, model: mjx.Model) -> tuple[Self, jax.Array]:
        data, new_rngs, relocation_occurred = self._relocate_food_items()

        parent_sim = self._parent_sim.update(data=data)
        parent_sim = parent_sim.step(model)

        return (
            self.update(
                _parent_sim=parent_sim,
                robots=self.robots.update(parent_sim.data),
                food_items=self.food_items.update(parent_sim.data),
                rngs_for_relocating_food=new_rngs
            ),
            relocation_occurred
        )

    def _collect_input_step(self, model: mjx.Model) -> Self:
        pheromone_x_idx, pheromone_y_idx = self._parent_sim.calc_nearest_pheromone_cell_indices(self.robots.positions)

        depths, _ = emit_rays(
            model, self.data, self.robots, self.consts.NUM_RAYS
        )

        pheromone_values = self.pheromone.get_gas(pheromone_x_idx, pheromone_y_idx)

        inputs = self.robot_inputs.update(
            ray=jnp.reciprocal(depths + 1e-6),
            pheromone=pheromone_values
        )

        return self.update(
            robot_inputs=inputs,
            _robot_pheromone_x_idx=pheromone_x_idx,
            _robot_pheromone_y_idx=pheromone_y_idx
        )

    @partial(jax.jit, inline=True, donate_argnames=("self",))
    def step(self, model: mjx.Model) -> Self:
        this = self._action_step()
        this, relocation_occurred = this._update_step(model)
        this = this._collect_input_step(model)

        # Calculate losses
        fr_losses = jax.vmap(
            BasicSimulatorWithEnv._calc_loss_between_food_and_robots,
            in_axes=(0, None, None),
            out_axes=0
        )(this.food_items.positions, this.robots.positions, this.consts)
        fn_losses = jax.vmap(
            BasicSimulatorWithEnv._calc_loss_between_food_and_nest,
            in_axes=(0, None),
            out_axes=0
        )(this.food_items.positions, this.consts)
        losses = fr_losses + fn_losses + this.loss_offset

        loss = jnp.sum(losses, keepdims=True)
        loss_offset = this.loss_offset + jnp.dot(relocation_occurred, losses)

        return this.update(
            loss=loss,
            _loss_offset=loss_offset
        )

    @partial(jax.jit, static_argnames=("n", "unroll"), inline=True, donate_argnames=("self",))
    def step_n(self, model: mjx.Model, n: int, unroll: int = 1) -> Self:
        def body_fn(carry: "BasicSimulatorWithEnv", _x) -> tuple["BasicSimulatorWithEnv", None]:
            new_carry = carry.step(model)
            return new_carry, None

        return jax.lax.scan(body_fn, self, length=n, unroll=unroll)[0]

    @partial(jax.jit, inline=True)
    def reset(self, model: mjx.Model) -> Self:
        this = self.update(_parent_sim=self._parent_sim.reset(model))
        return this.update(
            robots=this.robots.update(this.data),
            food_items=this.food_items.update(this.data),
            robot_inputs=this.robot_inputs.fill(0.0),
            robot_outputs=this.robot_outputs.fill(0.0),
            loss=jnp.zeros((1,), dtype=jnp.float32),
            _loss_offset=jnp.zeros((1,), dtype=jnp.float32),
        )

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        self._parent_sim.render(img_buf, camera, renderer)

    def evaluate(self) -> dict:
        return {"loss": self.loss}
