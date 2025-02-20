import numpy as np

rng = np.random.default_rng()


def robot_names(i: int):
    return {
        "body": f"bot{i}.body",
        "geom": f"bot{i}.geom",
        "joint_x": f"bot{i}.joint.slide_x",
        "joint_y": f"bot{i}.joint.slide_y",
        "joint_r": f"bot{i}.joint.hinge",
        "camera": f"bot{i}.camera",
        "act_x": f"bot{i}.act.horizontal",
        "act_y": f"bot{i}.act.vertical",
        "act_r": f"bot{i}.act.rotation",
        "center_site": f"bot{i}.site.center",
        "front_site": f"bot{i}.site.front",
        "velocimeter": f"bot{i}.sensor.vel",
    }

