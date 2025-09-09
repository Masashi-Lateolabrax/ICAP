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

    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        return self._parent_sim.get_pheromone(positions)

    def add_pheromone(self, positions: jax.Array, values: jax.Array) -> PheromoneField:
        return self._parent_sim.add_pheromone(positions, values)

    @staticmethod
    @partial(jax.jit, inline=True)
    def _check_food_in_nest(this: "BasicSimulatorWithEnv") -> jax.Array:
        distance_between_food_and_nest = jnp.linalg.norm(
            this.food_items.positions[:, :2] - this.consts.NEST_POSITION,
            axis=1
        )
        mask = distance_between_food_and_nest < this.consts.NEST_RADIUS
        return mask

    @staticmethod
    @partial(jax.jit, inline=True)
    def _generate_new_food_position(this: "BasicSimulatorWithEnv", rngs: jax.Array) -> jax.Array:
        key, rngs = jax.random.split(rngs)
        random_xy = jax.random.uniform(
            key,
            shape=(2,),
            minval=this.consts.NEST_RADIUS,
            maxval=jnp.array([this.consts.WORLD_WIDTH, this.consts.WORLD_HEIGHT]) * 0.5 - this.consts.FOOD_RADIUS
        )

        key, rngs = jax.random.split(rngs)
        sign = 2 * jax.random.randint(key, shape=(3,), minval=0, maxval=2).astype(jnp.float32) - 1
        random_xy = sign.at[:2].multiply(random_xy)
        random_xy = random_xy.at[2].set(2.0)

        return random_xy

    @staticmethod
    @partial(jax.jit, inline=True)
    def _relocate_food_items(
            this: "BasicSimulatorWithEnv"
    ) -> tuple["BasicSimulatorWithEnv", jax.Array]:
        relocation_happen = BasicSimulatorWithEnv._check_food_in_nest(this)

        def relocation_fn(i, sim: "BasicSimulatorWithEnv") -> "BasicSimulatorWithEnv":
            new_rngs, rngs = jax.random.split(this.rngs_for_relocating_food)
            new_position = BasicSimulatorWithEnv._generate_new_food_position(sim, rngs)
            new_data = sim.food_items.set_pos(this.data, i, new_position)
            return sim.update(
                data=new_data,
                rngs_for_relocating_food=new_rngs
            )

        this = jax.lax.fori_loop(
            0, this.food_items.positions.shape[0],
            lambda i, val: jax.lax.cond(
                relocation_happen[i],
                lambda x: relocation_fn(i, x),
                lambda x: x,
                val
            ),
            this
        )

        return this, relocation_happen

    @staticmethod
    @partial(jax.jit, inline=True)
    def _calc_loss_between_food_and_robots(
            food_position: jax.Array,
            robot_positions: jax.Array,
            const: Consts
    ) -> jax.Array:
        subs = (robot_positions[:, :2] - food_position[None, :2])
        distance = jnp.clip(
            jnp.linalg.norm(subs, axis=1) - const.OFFSET_FOOD_AND_ROBOT,
            a_min=0
        )
        return -jnp.sum(jnp.exp(-(distance ** 2) / const.SIGMA_FOOD_AND_ROBOT)) * const.GAIN_FOOD_AND_ROBOT

    @staticmethod
    @partial(jax.jit, inline=True)
    def _calc_loss_between_food_and_nest(
            food_position: jax.Array,
            nest_position: jax.Array,
            const: Consts
    ) -> jax.Array:
        distance = jnp.linalg.norm(food_position[:2] - nest_position[:2])
        distance = jnp.clip(distance - const.OFFSET_FOOD_AND_NEST, a_min=0)
        return -jnp.sum(jnp.exp(-(distance ** 2) / const.SIGMA_FOOD_AND_NEST)) * const.GAIN_FOOD_AND_NEST

    @staticmethod
    @partial(jax.jit, inline=True)
    def _step(this: "BasicSimulatorWithEnv", model: mjx.Model) -> "BasicSimulatorWithEnv":
        # First, apply robot outputs to the simulation
        this = this.update(
            data=this.robots.set_ctrl(this.data, this.robot_outputs.wheels),
            pheromone=this.add_pheromone(this.robots.positions, this.robot_outputs.pheromone)
        )

        # Step the basic simulator
        this = this.update(
            _parent_sim=this._parent_sim.step(model)
        )
        this = this.update(
            robots=this.robots.update(this.data),
            food_items=this.food_items.update(this.data),
        )

        # Emit rays and get inputs for robots
        depth_sensor: tuple[jax.Array, jax.Array] = emit_rays(
            model, this.data, this.robots, this.consts.NUM_RAYS
        )
        depths, _ = depth_sensor  # shape (num_robots, NUM_RAYS)
        inputs = this.robot_inputs.update(
            ray=jnp.reciprocal(depths + 1e-6)
        )

        # Get pheromone sensor values
        # pheromone_sensor = this.get_pheromone(this.robots.positions)
        # inputs = inputs.update(pheromone=pheromone_sensor)

        # Relocate food items if necessary
        this, relocation_occurred = BasicSimulatorWithEnv._relocate_food_items(this)

        # Calculate losses
        fr_losses = jax.vmap(lambda x: BasicSimulatorWithEnv._calc_loss_between_food_and_robots(
            x, this.robots.positions, this.consts
        ))(this.food_items.positions)
        fn_losses = jax.vmap(lambda x: BasicSimulatorWithEnv._calc_loss_between_food_and_nest(
            x, this.consts.NEST_POSITION, this.consts
        ))(this.food_items.positions)
        losses = fr_losses + fn_losses + this.loss_offset

        loss = jnp.sum(losses, keepdims=True)
        loss_offset = this.loss_offset + jnp.dot(relocation_occurred, losses)

        return this.update(
            robot_inputs=inputs,
            loss=loss,
            _loss_offset=loss_offset
        )

    def step(self, model: mjx.Model) -> Self:
        return BasicSimulatorWithEnv._step(self, model)

    @staticmethod
    @partial(jax.jit, static_argnames=("n",), inline=True)
    def _step_n(simulator: "BasicSimulatorWithEnv", model: mjx.Model, n: int) -> "BasicSimulatorWithEnv":
        def body_fn(_i, sim: "BasicSimulatorWithEnv"):
            return BasicSimulatorWithEnv._step(sim, model)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, simulator)
        return new_simulator

    def step_n(self, model: mjx.Model, n: int) -> Self:
        return BasicSimulatorWithEnv._step_n(self, model, n)

    @partial(jax.jit, inline=True)
    def reset(self, model: mjx.Model) -> Self:
        this = self.update(_parent_sim=self._parent_sim.reset(model))

        new_robots = self.robots.update(this.data)
        new_food_items = self.food_items.update(this.data)

        new_inputs = this.robot_inputs.fill(0.0)
        new_outputs = this.robot_outputs.fill(0.0)

        return this.update(
            robots=new_robots,
            food_items=new_food_items,
            robot_inputs=new_inputs,
            robot_outputs=new_outputs,
            loss=jnp.zeros((1,), dtype=jnp.float32),
            _loss_offset=jnp.zeros((1,), dtype=jnp.float32),
        )

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        self._parent_sim.render(img_buf, camera, renderer)

    def evaluate(self) -> dict:
        return {"loss": self.loss}
