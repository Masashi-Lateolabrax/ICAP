import abc


class MuJoCoXMLGenerator:
    class ToString(metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def to_string(self) -> str:
            raise NotImplementedError()

    class MuJoCoCompiler(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoOption(ToString):
        def __init__(self, attributes):
            self.attributes = attributes

        def to_string(self) -> str:
            xml = "<option "
            for k, v in self.attributes.items():
                xml += f"{k}=\"{v}\" "
            xml += ">\n"
            xml += "</option>\n"
            return xml

    class MuJoCoSize(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoVisual(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoStatistic(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoDefault(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoCustom(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoAsset(ToString):
        def __init__(self):
            self.texture_attributes = []

        def add_texture(self, attributes):
            self.texture_attributes.append(attributes)

        def to_string(self) -> str:
            xml = "<asset>\n"
            for ta in self.texture_attributes:
                xml += "<texture "
                for k, v in ta.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"
            xml += "</asset>\n"
            return xml

    class MuJoCoBody(ToString):
        def __init__(self, is_world: bool, attributes):
            self._worldbody = is_world
            self._name = None
            self._attributes = attributes
            self._geoms = []
            self._site = []
            self._joint = []
            self._child_body: list[MuJoCoXMLGenerator.MuJoCoBody] = []

            if self._worldbody:
                self._name = "worldbody"
            elif "name" in attributes:
                self._name = attributes["name"]

        def add_body(self, attributes):
            child_body = MuJoCoXMLGenerator.MuJoCoBody(False, attributes)
            self._child_body.append(child_body)
            return child_body

        def add_geom(self, attributes):
            self._geoms.append(attributes)

        def add_site(self, attributes):
            self._site.append(attributes)

        def add_joint(self, attributes):
            self._joint.append(attributes)

        def add_freejoint(self):
            self.add_joint({"type": "free", "stiffness": "0", "damping": "0", "frictionloss": "0", "armature": "0"})

        def to_string(self) -> str:
            if self._worldbody:
                xml = "<worldbody>\n"
            else:
                xml = "<body "
                for k, v in self._attributes.items():
                    xml += f"{k}=\"{v}\" "
                xml += ">\n"

            for j in self._joint:
                xml += "<joint "
                for k, v in j.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            for g in self._geoms:
                xml += "<geom "
                for k, v in g.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            for g in self._site:
                xml += "<site "
                for k, v in g.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            for child in self._child_body:
                xml += child.to_string()

            if self._worldbody:
                xml += "</worldbody>\n"
            else:
                xml += "</body>\n"

            return xml

    class MuJoCoContact(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoEquality(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoTendon(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoActuator(ToString):
        def __init__(self):
            self.velocity = []

        def add_velocity(self, attributes):
            self.velocity.append(attributes)

        def to_string(self) -> str:
            xml = "<actuator>\n"
            for va in self.velocity:
                xml += "<velocity "
                for k, v in va.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"
            xml += "</actuator>\n"
            return xml

    class MuJoCoSensor(ToString):
        def __init__(self):
            self.sensor = []

        def add_velocimeter(self, attributes):
            element = "<velocimeter "
            for k, v in attributes.items():
                element += f"{k}=\"{v}\" "
            element += "/>"
            self.sensor.append(element)

        def to_string(self) -> str:
            xml = "<sensor>\n"
            for e in self.sensor:
                xml += e + "\n"
            xml += "</sensor>\n"
            return xml

    class MuJoCoKeyframe(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.elements = list([MuJoCoXMLGenerator.MuJoCoBody(True, {})])

    def add_compiler(self) -> MuJoCoCompiler:
        pass

    def add_option(self, attributes) -> MuJoCoOption:
        element = MuJoCoXMLGenerator.MuJoCoOption(attributes)
        self.elements.append(element)
        return element

    def add_size(self) -> MuJoCoSize:
        pass

    def add_visual(self) -> MuJoCoVisual:
        pass

    def add_statistic(self) -> MuJoCoStatistic:
        pass

    def add_default(self) -> MuJoCoDefault:
        pass

    def add_custom(self) -> MuJoCoCustom:
        pass

    def add_asset(self) -> MuJoCoAsset:
        element = MuJoCoXMLGenerator.MuJoCoAsset()
        self.elements.append(element)
        return element

    def get_body(self, name: str = "worldbody") -> MuJoCoBody:
        find = None
        index = 0
        candidate = [e for e in self.elements if type(e) == MuJoCoXMLGenerator.MuJoCoBody]
        while index < len(candidate):
            if candidate[index]._name == name:
                find = candidate[index]
                break
            for new_c in candidate[index]._child_body:
                candidate.append(new_c)
            index += 1
        return find

    def add_contact(self) -> MuJoCoContact:
        pass

    def add_equality(self) -> MuJoCoEquality:
        pass

    def add_tendon(self) -> MuJoCoTendon:
        pass

    def add_actuator(self) -> MuJoCoActuator:
        element = MuJoCoXMLGenerator.MuJoCoActuator()
        self.elements.append(element)
        return element

    def add_sensor(self) -> MuJoCoSensor:
        element = MuJoCoXMLGenerator.MuJoCoSensor()
        self.elements.append(element)
        return element

    def add_keyframe(self) -> MuJoCoKeyframe:
        pass

    def generate(self):
        xml = f"<mujoco model=\"{self.model_name}\">\n"
        for e in self.elements:
            xml += e.to_string()
        xml += "</mujoco>\n"
        return xml
