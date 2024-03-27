import time

import mujoco
import mujoco.viewer

import mujoco_xml_generator as mjc_gen
from mujoco_xml_generator import common as mjc_cmn
from mujoco_xml_generator import WorldBody, Body, body
from mujoco_xml_generator import Default, default


def gen_xml() -> str:
    generator = mjc_gen.Generator().add_children([
        Default("goal").add_children([
            default.Geom(
                type_=mjc_cmn.GeomType.BOX, size=(0.20, 0.20, 0.01), rgba=(0, 1, 0, 1), contype=2, conaffinity=2
            )
        ]),

        WorldBody().add_children([
            body.Geom(
                type_=mjc_cmn.GeomType.PLANE, size=(10, 10, 1), rgba=(1, 1, 1, 0.5)
            ),

            body.Geom(class_="goal", pos=(1, 0, 0.01)),
            body.Geom(class_="goal", pos=(0, 1, 0.01)),
            body.Geom(class_="goal", pos=(1, 1, 0.01)),
            body.Geom(class_="goal", pos=(-1, 0, 0.01)),
            body.Geom(class_="goal", pos=(0, -1, 0.01)),

            Body(
                name="bot1", pos=(0, 0, 0.06)
            ).add_children([
                body.Joint(type_=body.Joint.JointType.FREE),
                body.Geom(type_=mjc_cmn.GeomType.CYLINDER, size=(0.15, 0.05), rgba=(1, 1, 0, 1)),
                body.Camera("bot1cam", orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90), pos=(0, 0.16, 0))
            ])
        ])
    ])
    return generator.build()


def calc_robot_sight(model, bot_name, start, end, div):
    import numpy

    mat = numpy.zeros((3, 3))
    mat[2, 2] = 1.0
    step = (end - start) / div
    res = numpy.zeros(div * 2)
    start += step * 0.8

    offset = numpy.array([0, 0, 4.5])

    for i in range(0, div):
        theta = start + step * i
        mat[0, 0] = numpy.cos(theta)
        mat[0, 1] = numpy.sin(theta)
        mat[1, 0] = -numpy.sin(theta)
        mat[1, 1] = numpy.cos(theta)
        geom_name, distance = model.calc_ray(
            robot.pos + offset,
            numpy.dot(mat, robot.direction),
            exclude_id=robot.id
        )

        if geom_name is None:
            res[i * 2 + 1] = 0.0
        elif geom_name.find("robot") != -1:
            res[i * 2 + 1] = 0.3
        elif geom_name.find("feed") != -1:
            res[i * 2 + 1] = 0.6
        elif geom_name.find("obstacle") != -1:
            res[i * 2 + 1] = 0.9
        else:
            res[i * 2 + 1] = 0.0

        half_width = 200
        if distance >= 0:
            res[i * 2] = 1.0 - numpy.tanh(numpy.arctanh(0.5) * distance / half_width)
        else:
            res[i * 2] = 0.0

    return res


def main():
    model = mujoco.MjModel.from_xml_string(xml=gen_xml())
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == '__main__':
    main()
