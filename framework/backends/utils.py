from functools import partial

import mujoco
from mujoco import mjx

import numpy as np
from icecream import ic

import jax
import jax.numpy as jnp

from ..prelude import *


def render(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        render_shape: tuple[int, int],
        max_geom: int,
        img_buf: np.ndarray,
        pos: tuple[float, float, float],
        lookat: tuple[float, float, float]
):
    camera = mujoco.MjvCamera()
    pos = np.array(pos)
    lookat = np.array(lookat)
    sub = pos - lookat

    camera.lookat[:] = lookat
    camera.distance = np.linalg.norm(sub)
    camera.azimuth = np.arctan2(
        sub[1], sub[0]
    ) * 180 / mujoco.mjPI + 180
    camera.elevation = -np.arcsin(
        sub[2] / camera.distance
    ) * 180 / mujoco.mjPI

    try:
        with mujoco.Renderer(
                model, width=render_shape[0], height=render_shape[1], max_geom=max_geom
        ) as renderer:
            renderer.update_scene(data, camera)
            renderer.render(out=img_buf)

    except Exception as e:
        ic("MuJoCo render error:", e)
        img_buf.fill(0)


@partial(jax.jit, static_argnames=["body_id"], inline=True)
def _jitted_ray_fn(
        vec: jax.Array, model: mjx.Model, data: mjx.Data, robot_pos: jax.Array, body_id: int
) -> tuple[jax.Array, jax.Array]:
    return mjx.ray(model, data, robot_pos, vec, (), True, body_id)


@partial(jax.jit, static_argnames=["body_id", "num_rays"], inline=True)
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


@partial(jax.jit, static_argnames=["body_ids", "num_rays"], inline=True)
def emit_rays_static(
        model: mjx.Model,
        data: mjx.Data,
        positions: jax.Array,
        xdirections: jax.Array,
        body_ids: tuple[int, ...],
        num_rays: int
) -> tuple[jax.Array, jax.Array]:  # shape (num_robots, num_rays), (num_robots, num_rays)
    results = []
    for i, body_id in enumerate(body_ids):
        result = _emit_n_rays(
            model, data,
            positions[i], xdirections[i],
            body_id, num_rays
        )
        results.append(result)

    dists = jnp.stack([r[0] for r in results])
    ids = jnp.stack([r[1] for r in results])
    return dists, ids


@partial(jax.jit, static_argnames=["num_rays"], inline=True)
def emit_rays(
        model: mjx.Model,
        data: mjx.Data,
        robots: BatchedRobots,
        num_rays: int
) -> tuple[jax.Array, jax.Array]:  # shape (num_robots, NUM_RAYS), (num_robots, NUM_RAYS)
    return emit_rays_static(
        model, data,
        robots.positions, robots.xdirections,
        robots.ids.body_ids, num_rays
    )
