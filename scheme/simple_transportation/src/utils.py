def robot_names(i: int):
    return {
        "body": f"bot{i}.body",
        "geom": f"bot{i}.geom",
        "x_joint": f"bot{i}.joint.slide_x",
        "y_joint": f"bot{i}.joint.slide_y",
        "r_joint": f"bot{i}.joint.hinge",
        "camera": f"bot{i}.camera",
        "x_act": f"bot{i}.act.pos_x",
        "y_act": f"bot{i}.act.pos_y",
        "r_act": f"bot{i}.joint.hinge"
    }
