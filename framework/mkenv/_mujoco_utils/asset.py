from enum import Enum

import mujoco


class MeshContentType(Enum):
    MODEL = "model/vnd.mujoco.mesh"
    STL = "model/stl"
    OBJ = "model/obj"


def add_mesh_in_asset(
        spec: mujoco.MjSpec,
        name: str,
        file: str = None,
        content_type: MeshContentType = None,
        inertia: mujoco.mjtMeshInertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_LEGACY,
        refpos: tuple[float, float, float] = (0, 0, 0),
        refquat: tuple[float, float, float, float] = (1, 0, 0, 0),
) -> mujoco._specs.MjsGeom:
    mesh: mujoco.MjsMesh = spec.add_mesh()

    if name:
        mesh.name = name
    if file:
        mesh.file = file
    if content_type:
        mesh.content_type = content_type.value

    mesh.inertia = inertia
    mesh.refpos[:] = refpos
    mesh.refquat[:] = refquat

    return mesh
