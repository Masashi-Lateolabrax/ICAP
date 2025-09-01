from functools import partial

import mujoco
import mujoco.mjx as mjx

import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass as jax_dataclass

from ..prelude import *
from .basic_environment import generate_mjspec
from .jaxable_basic import BasicSimulator


@partial(jax.jit, static_argnames=["body_id"])
def _jitted_ray_fn(
        vec: jax.Array, model: mjx.Model, data: mjx.Data, robot_pos: jax.Array, body_id: int
) -> tuple[jax.Array, jax.Array]:
    return mjx.ray(model, data, robot_pos, vec, (), True, body_id)


@partial(jax.jit, static_argnames=["body_id", "num_rays"])
def _emit_n_rays(
        model: mjx.Model,
        data: mjx.Data,
        robot_pos: jax.Array,
        robot_xdir: jax.Array,
        body_id: int,
        num_rays: int
) -> tuple[jax.Array, jax.Array]:  # shape (num_rays,), (num_rays,)
    def body_fn(vec) -> tuple[jax.Array, jax.Array]:
        return _jitted_ray_fn(vec, model, data, robot_pos, body_id)

    delta_angle = 2 * jnp.pi / num_rays
    angles = jnp.arange(num_rays) * delta_angle  # Shape (num_rays,)
    cos = jnp.cos(angles)  # Shape (num_rays,)
    sin = jnp.sin(angles)  # Shape (num_rays,)
    horizontal_elements = cos * robot_xdir[0] - sin * robot_xdir[1]  # Shape (num_rays,)
    vertical_elements = sin * robot_xdir[0] + cos * robot_xdir[1]  # Shape (num_rays,)
    rotated_dirs = jnp.stack(
        [horizontal_elements, vertical_elements, jnp.zeros(num_rays)],
        axis=1
    )  # Shape (num_rays, 3)

    dists, ids = jax.vmap(body_fn)(rotated_dirs)
    return dists, ids


_EMIT_RAYS_FUNCTIONS = []
_REGISTERED_ROBOT_IDS = set()


def create_emit_rays_functions(num_rays: int, robots: BatchedRobots):
    global _EMIT_RAYS_FUNCTIONS, _REGISTERED_ROBOT_IDS

    body_ids = robots.ids.body_ids.tolist()

    for id_ in body_ids:
        if id_ in _REGISTERED_ROBOT_IDS:
            continue

        @partial(jax.jit, static_argnames=["body_id_", "num_rays_"])
        def emit_rays_fn(model, data, pos, xdir, body_id_=id_, num_rays_=num_rays):
            return _emit_n_rays(model, data, pos, xdir, body_id_, num_rays_)

        _REGISTERED_ROBOT_IDS.add(id_)
        _EMIT_RAYS_FUNCTIONS.append(emit_rays_fn)


def emit_rays(
        model: mjx.Model,
        data: mjx.Data,
        robots: BatchedRobots,
) -> tuple[jax.Array, jax.Array]:  # shape (num_robots, NUM_RAYS), (num_robots, NUM_RAYS)
    def body_fn(pos_, xdir_, func_idx_) -> tuple[jax.Array, jax.Array]:  # shape (NUM_RAYS,), (NUM_RAYS,)
        return jax.lax.switch(func_idx_, _EMIT_RAYS_FUNCTIONS, model, data, pos_, xdir_)

    positions = robots.positions
    xdirections = robots.xdirections
    func_idx = jnp.arange(robots.num_robots)
    return jax.vmap(body_fn)(positions, xdirections, func_idx)


@jax_dataclass
class Consts:
    WORLD_WIDTH: float
    WORLD_HEIGHT: float

    NEST_POSITION: jax.Array
    NEST_RADIUS: float

    FOOD_RADIUS: float

    OFFSET_FOOD_AND_ROBOT: float
    SIGMA_FOOD_AND_ROBOT: float
    GAIN_FOOD_AND_ROBOT: float

    OFFSET_FOOD_AND_NEST: float
    SIGMA_FOOD_AND_NEST: float
    GAIN_FOOD_AND_NEST: float


@jax_dataclass
class BasicSimulatorWithEnv:
    _basic_sim: BasicSimulator

    robots: BatchedRobots
    robot_inputs: jax.Array  # shape (num_robots, NUM_RAYS)
    food_items: BatchedFood
    counter_for_relocation: int
    rngs_for_relocating_food: jax.Array

    consts: Consts

    @property
    def model(self) -> mjx.Model:
        return self._basic_sim.model

    @property
    def data(self) -> mjx.Data:
        return self._basic_sim.data

    def update(
            self,
            model: mjx.Model = None,
            data: mjx.Data = None,

            robots: BatchedRobots = None,
            robot_inputs: jax.Array = None,
            food_items: BatchedFood = None,
            counter_for_relocation: int = None,
            rngs_for_relocating_food: jax.Array = None,
    ) -> 'BasicSimulatorWithEnv':
        parent_kwargs = {
            "model": model,
            "data": data,
        }
        basic_sim = self._basic_sim.update(**parent_kwargs)

        this_kwargs = {
            "_basic_sim": basic_sim,
            "robots": robots,
            "robot_inputs": robot_inputs,
            "food_items": food_items,
            "counter_for_relocation": counter_for_relocation,
            "rngs_for_relocating_food": rngs_for_relocating_food,
        }
        kwargs = {k: v for k, v in this_kwargs.items() if v is not None}
        if not kwargs:
            return self

        return self.replace(**kwargs)

    @classmethod
    def new(cls, settings: Settings, rngs: jax.Array) -> 'BasicSimulatorWithEnv':
        mj_spec, nest_spec, robot_specs, food_specs = generate_mjspec(settings)
        basic_sim = BasicSimulator.new(mj_spec, settings)

        batched_robot_id = BatchedRobotIDs.from_specs(basic_sim.model, robot_specs)
        batched_food_id = BatchedFoodIDs.from_specs(basic_sim.model, food_specs)

        distance_between_wheels = settings.Robot.DISTANCE_BETWEEN_WHEELS
        max_speed = settings.Robot.MAX_SPEED
        robots = BatchedRobots.new(basic_sim.data, batched_robot_id, distance_between_wheels, max_speed)

        food_items: BatchedFood = BatchedFood.new(basic_sim.data, batched_food_id)

        create_emit_rays_functions(
            settings.Robot.NUM_RAYS, robots
        )
        robot_inputs = jnp.zeros((robots.num_robots, settings.Robot.NUM_RAYS))

        return cls(
            _basic_sim=basic_sim,

            robots=robots,
            robot_inputs=robot_inputs,
            food_items=food_items,
            counter_for_relocation=0,
            rngs_for_relocating_food=rngs,

            consts=Consts(
                WORLD_WIDTH=settings.Simulation.WORLD_WIDTH,
                WORLD_HEIGHT=settings.Simulation.WORLD_HEIGHT,
                NEST_POSITION=settings.Nest.POSITION.as_array(),
                NEST_RADIUS=settings.Nest.RADIUS,
                FOOD_RADIUS=settings.Food.RADIUS,
                OFFSET_FOOD_AND_ROBOT=settings.Loss.OFFSET_FOOD_AND_ROBOT,
                SIGMA_FOOD_AND_ROBOT=settings.Loss.SIGMA_FOOD_AND_ROBOT,
                GAIN_FOOD_AND_ROBOT=settings.Loss.GAIN_FOOD_AND_ROBOT,
                OFFSET_FOOD_AND_NEST=settings.Loss.OFFSET_FOOD_AND_NEST,
                SIGMA_FOOD_AND_NEST=settings.Loss.SIGMA_FOOD_AND_NEST,
                GAIN_FOOD_AND_NEST=settings.Loss.GAIN_FOOD_AND_NEST,
            )
        )

    def get_pheromone(self, positions: jax.Array) -> jax.Array:
        return self._basic_sim.get_pheromone(positions)

    def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> 'BasicSimulatorWithEnv':
        new_basic_sim = self._basic_sim.add_pheromone(positions, amounts)
        return self.replace(_basic_sim=new_basic_sim)

    @staticmethod
    @jax.jit
    def _check_food_in_nest(this: "BasicSimulatorWithEnv") -> jax.Array:
        distance_between_food_and_nest = jnp.linalg.norm(
            this.food_items.positions[:, :2] - this.NEST_POSITION,
            axis=1
        )
        mask = distance_between_food_and_nest < this.NEST_RADIUS
        return mask

    @staticmethod
    @jax.jit
    def _generate_new_food_position(this: "BasicSimulatorWithEnv", rngs: jax.Array) -> jax.Array:
        key, rngs = jax.random.split(this.rngs_for_relocating_food)
        random_xy = jax.random.uniform(
            key,
            shape=(2,),
            minval=this.NEST_RADIUS,
            maxval=jnp.array([this.WORLD_WIDTH, this.WORLD_HEIGHT]) - this.FOOD_RADIUS
        )

        key, rngs = jax.random.split(rngs)
        sign = 2 * jax.random.randint(key, shape=(3,), minval=0, maxval=2) - 1
        random_xy = sign.at[:2].multiply(random_xy)
        random_xy = random_xy.at[2].set(5.0)

        return random_xy

    @staticmethod
    @jax.jit
    def _relocate_food_item(
            this: "BasicSimulatorWithEnv"
    ) -> tuple["BasicSimulatorWithEnv", int, jax.Array]:
        idx = new_counter = (this.counter_for_relocation + 1) % this.food_items.positions.shape[0]

        position = this.food_items.positions[idx, :]
        distance = jnp.linalg.norm(position[:2] - this.NEST_POSITION)
        has_to_relocate = distance < this.NEST_RADIUS

        def body_fn(sim: "BasicSimulatorWithEnv") -> "BasicSimulatorWithEnv":
            new_rngs, rngs = jax.random.split(this.rngs_for_relocating_food)
            new_position = BasicSimulatorWithEnv._generate_new_food_position(sim, rngs)
            new_data = sim.food_items.set_pos(this.data, idx, new_position)
            return sim.update(
                data=new_data,
                rngs_for_relocating_food=new_rngs
            )

        this = jax.lax.cond(
            has_to_relocate,
            lambda x: body_fn(x),
            lambda x: x,
            this
        )

        this = this.update(
            counter_for_relocation=new_counter,
        )
        return this, idx, has_to_relocate

    @staticmethod
    @jax.jit
    def _step(this: "BasicSimulatorWithEnv", dt) -> "BasicSimulatorWithEnv":
        this = this.replace(_basic_sim=this._basic_sim.step(dt))

        this = this.update(
            robots=this.robots.update(this.data),
            food_items=this.food_items.update(this.data),
        )

        depth_sensor: tuple[jax.Array, jax.Array] = emit_rays(
            this.model, this.data, this.robots
        )
        depths, _ = depth_sensor  # shape (num_robots, NUM_RAYS)
        inputs = jnp.reciprocal(depths + 1e-6)

        this, idx, relocation_occurred = BasicSimulatorWithEnv._relocate_food_item(this)

        return this.update(robot_inputs=inputs)

    def step(self, dt: float) -> 'BasicSimulatorWithEnv':
        return BasicSimulatorWithEnv._step(self, dt)

    @staticmethod
    @jax.jit
    def _step_n(simulator: "BasicSimulatorWithEnv", n: int, dt: float) -> "BasicSimulatorWithEnv":
        def body_fn(_i, sim: "BasicSimulatorWithEnv"):
            return BasicSimulatorWithEnv._step(sim, dt)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, simulator)
        return new_simulator

    def step_n(self, n: int, dt: float) -> 'BasicSimulatorWithEnv':
        return BasicSimulatorWithEnv._step_n(self, n, dt)

    def render(
            self,
            mj_model: mujoco.MjModel,
            img_buf: np.ndarray,
            pos: tuple[float, float, float],
            lookat: tuple[float, float, float],
            max_geom=100,
            max_pheromone=1.0
    ):
        self._basic_sim.render(mj_model, img_buf, pos, lookat, max_geom, max_pheromone)

    def reset(self, rngs: jax.Array = None) -> 'BasicSimulatorWithEnv':
        new_basic_sim = self._basic_sim.reset()
        new_robots = self.robots.update(new_basic_sim.data)
        new_food_items = self.food_items.update(new_basic_sim.data)

        this = self.replace(_basic_sim=new_basic_sim)
        return this.update(
            robots=new_robots,
            food_items=new_food_items,
            rngs_for_relocating_food=rngs
        )
