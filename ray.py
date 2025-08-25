# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for ray interesection testing."""

from typing import Sequence, Tuple

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import math
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import GeomType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np


def _ray_quad(
    a: jax.Array, b: jax.Array, c: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Returns two solutions for quadratic: a*x^2 + 2*b*x + c = 0."""
  det = b * b - a * c
  det_2 = jp.sqrt(det)

  x0, x1 = (-b - det_2) / a, (-b + det_2) / a
  x0 = jp.where((det < mujoco.mjMINVAL) | (x0 < 0), jp.inf, x0)
  x1 = jp.where((det < mujoco.mjMINVAL) | (x1 < 0), jp.inf, x1)

  return x0, x1


def _ray_plane(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a plane."""
  if size.shape != (3,) or pnt.shape != (3,) or vec.shape != (3,):
    raise ValueError(
        f'Expected inputs with shape (3,), got size: {size.shape}, pnt:'
        f' {pnt.shape}, vec: {vec.shape}'
    )

  # Collision point: p = pnt + t*vec
  # Plane equation: point[i]*normal[i] = size[i]

  # test if the ray is parallel to the surface
  # collision if |vec[i]*size[i]| < mjMINVAL

  # solve for collision distance t=d/vec_n where:
  #   d = size[i] - pnt[i]
  #   vec_n = vec[i]*normal[i]
  # and normal[i] is the sign of size[i] to handle double-sided planes

  d = size - jp.where(size >= 0, pnt, -pnt)
  vec_n = jp.where(size >= 0, vec, -vec)

  collision_dist = jp.where(
      jp.abs(vec_n) < mujoco.mjMINVAL, jp.inf, jp.where(d < 0, jp.inf, d / vec_n)
  )

  # find closest collision point
  return jp.min(collision_dist)


def _ray_sphere(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a sphere."""
  if size.shape != (3,) or pnt.shape != (3,) or vec.shape != (3,):
    raise ValueError(
        f'Expected inputs with shape (3,), got size: {size.shape}, pnt:'
        f' {pnt.shape}, vec: {vec.shape}'
    )

  r = size[0]
  a = jp.sum(vec * vec)
  b = jp.sum(pnt * vec)
  c = jp.sum(pnt * pnt) - r * r

  x0, x1 = _ray_quad(a, b, c)

  return jp.min(jp.array([x0, x1]))


def _ray_capsule(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a capsule."""
  if size.shape != (3,) or pnt.shape != (3,) or vec.shape != (3,):
    raise ValueError(
        f'Expected inputs with shape (3,), got size: {size.shape}, pnt:'
        f' {pnt.shape}, vec: {vec.shape}'
    )

  r = size[0]
  z = jp.abs(size[2])

  # collision distance is the shortest of:
  # 1. sphere centered on (0, 0, z)
  # 2. sphere centered on (0, 0, -z)
  # 3. cylinder with infinite height

  # collision with cylinder: pnt[0:1] + t*vec[0:1], r = size[0]
  pnt_01 = pnt[:2]
  vec_01 = vec[:2]

  a = jp.sum(vec_01 * vec_01)
  b = jp.sum(pnt_01 * vec_01)
  c = jp.sum(pnt_01 * pnt_01) - r * r

  x0, x1 = _ray_quad(a, b, c)

  # check if the collision point is within the cylinder
  z0, z1 = pnt[2] + x0 * vec[2], pnt[2] + x1 * vec[2]
  cyl_0 = jp.where(jp.abs(z0) < z, x0, jp.inf)
  cyl_1 = jp.where(jp.abs(z1) < z, x1, jp.inf)

  # collision with upper cap
  pnt_upper = pnt - jp.array([0.0, 0.0, z])
  cap_upper = _ray_sphere(jp.array([r, r, r]), pnt_upper, vec)

  # collision with lower cap
  pnt_lower = pnt - jp.array([0.0, 0.0, -z])
  cap_lower = _ray_sphere(jp.array([r, r, r]), pnt_lower, vec)

  return jp.min(jp.array([cyl_0, cyl_1, cap_upper, cap_lower]))


def _ray_box(
    size: jax.Array,
    pnt: jax.Array,
    vec: jax.Array,
) -> jax.Array:
  """Returns the distance at which a ray intersects with a box."""
  if size.shape != (3,) or pnt.shape != (3,) or vec.shape != (3,):
    raise ValueError(
        f'Expected inputs with shape (3,), got size: {size.shape}, pnt:'
        f' {pnt.shape}, vec: {vec.shape}'
    )

  # if the ray starts inside the box, we want to trace to the edge
  inside = jp.all(jp.abs(pnt) < size)

  # collision point: p = pnt + t*vec
  # box faces: |x| < size[0], |y| < size[1], |z| < size[2]

  # solve for all 6 faces
  # ray to positive faces: pnt + t*vec = size -> t = (size - pnt) / vec
  # ray to negative faces: pnt + t*vec = -size -> t = (-size - pnt) / vec

  t_pos = (size - pnt) / vec
  t_neg = (-size - pnt) / vec

  # check collision at faces by testing the other dims
  def collision_at_face(t: jax.Array, ax: int) -> jax.Array:
    pos = pnt + t * vec
    p_ax = jp.roll(pos, -ax)[:2]  # pos except position ax
    s_ax = jp.roll(size, -ax)[:2]  # size except position ax
    return jp.where(jp.all(jp.abs(p_ax) <= s_ax), t, jp.inf)

  t_hit = jp.concatenate([
      jp.array([collision_at_face(t_pos[i], i) for i in range(3)]),
      jp.array([collision_at_face(t_neg[i], i) for i in range(3)]),
  ])

  # filter collisions that go backwards
  t_hit = jp.where(t_hit <= 0, jp.inf, t_hit)

  if inside:
    return jp.where(jp.all(jp.isinf(t_hit)), 0.0, jp.min(t_hit))
  else:
    return jp.min(t_hit)


def ray(
    m: Model,
    d: Data,
    pnt: jax.Array,
    vec: jax.Array,
    geom_group: Sequence[int] = (),
    flg_static: bool = True,
    bodyexclude: int = -1,
    geomexclude: int = -1,
) -> Tuple[jax.Array, int, jax.Array]:
  """Intersect ray with nearest geom, get distance and 3D coordinates of point.

  This function casts a ray into the scene and returns the distance to the
  nearest collision, the id of the geom that was hit, and the coordinates
  of the collision point.

  Note that this function does not perform collision detection between the
  ray and any geom that is attached to the excluded body.

  Args:
    m: The MuJoCo model.
    d: The MuJoCo data.
    pnt: The origin of the ray in global coordinates.
    vec: The direction of the ray in global coordinates.
    geom_group: Only geoms in this group will be checked. If empty, no group
      filtering is done.
    flg_static: Whether to check static geoms.
    bodyexclude: Body whose geoms will be excluded from the ray cast.
    geomexclude: Geom that will be excluded from the ray cast.

  Returns:
    The distance to the nearest collision, the id of the geom that was hit,
    and the coordinates of the collision point. If no geom was hit, returns
    (-1, -1, pnt).
  """
  # defaults for no collision case
  geom_hit = -1
  collision_pos = jp.array([0., 0., 0.])
  collision_dist = -1.0

  for geom_id in range(m.ngeom):
    # Check for exclusions
    if geom_id == geomexclude:
      continue

    if (bodyexclude >= 0) and (m.geom_bodyid[geom_id] == bodyexclude):
      continue

    # Check group filtering
    if geom_group and (m.geom_group[geom_id] not in geom_group):
      continue

    # Check if it's a static geom
    if not flg_static and (m.body_parentid[m.geom_bodyid[geom_id]] == 0):
      continue

    # Get collision distance
    geom_pos = d.geom_xpos[geom_id]
    geom_mat = d.geom_xmat[geom_id].reshape((3, 3))

    # Transform ray to geom-local coordinates
    pnt_local = math.rotate(pnt - geom_pos, geom_mat.T)
    vec_local = math.rotate(vec, geom_mat.T)

    geom_type = m.geom_type[geom_id]
    geom_size = m.geom_size[geom_id]

    if geom_type == GeomType.mjGEOM_PLANE:
      dist = _ray_plane(geom_size, pnt_local, vec_local)
    elif geom_type == GeomType.mjGEOM_SPHERE:
      dist = _ray_sphere(geom_size, pnt_local, vec_local)
    elif geom_type == GeomType.mjGEOM_CAPSULE:
      dist = _ray_capsule(geom_size, pnt_local, vec_local)
    elif geom_type == GeomType.mjGEOM_BOX:
      dist = _ray_box(geom_size, pnt_local, vec_local)
    else:
      # Unsupported geom type, skip
      continue

    # Update collision if this is closer
    is_closer = (collision_dist == -1.0) or (
        (dist < collision_dist) and not jp.isinf(dist)
    )
    geom_hit = jp.where(is_closer, geom_id, geom_hit)
    collision_dist = jp.where(is_closer, dist, collision_dist)
    collision_pos_candidate = pnt + dist * vec
    collision_pos = jp.where(is_closer, collision_pos_candidate, collision_pos)

  return collision_dist, geom_hit, collision_pos