class RobotNameTable:
    def __init__(self, id_):
        self.BODY = f"bot{id_}.body"
        self.GEOM = f"bot{id_}.geom"
        self.JOINT_X = f"bot{id_}.joint.slide_x"
        self.JOINT_Y = f"bot{id_}.joint.slide_y"
        self.JOINT_R = f"bot{id_}.joint.hinge"
        self.CAMERA = f"bot{id_}.camera"
        self.ACT_X = f"bot{id_}.act.horizontal"
        self.ACT_Y = f"bot{id_}.act.vertical"
        self.ACT_R = f"bot{id_}.act.rotation"
        self.CENTER_SITE = f"bot{id_}.site.center"
        self.FRONT_SITE = f"bot{id_}.site.front"
        self.VELOCIMETER = f"bot{id_}.sensor.vel"