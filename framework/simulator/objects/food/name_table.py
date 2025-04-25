class FoodNameTable:
    def __init__(self, id_):
        self.BODY = f"food{id_}.body"
        self.JOINT_X = f"food{id_}.joint.x"
        self.JOINT_Y = f"food{id_}.joint.y"
        self.CENTER_SITE = f"food{id_}.site.center"
        self.VELOCIMETER = f"food{id_}.sensor.velocity"
