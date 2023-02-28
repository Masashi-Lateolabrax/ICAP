import abc
import enum


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
            self._headlight = None

        def add_headlight(self, attributes):
            self._headlight = attributes

        def to_string(self) -> str:
            xml = "<visual>"

            if self._headlight is not None:
                xml += "<headlight "
                for k, v in self._headlight.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            xml += "</visual>\n"
            return xml

    class MuJoCoStatistic(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoDefault(ToString):
        def __init__(self, class_name: str):
            self._class_name = class_name
            self._children: list[MuJoCoXMLGenerator.MuJoCoDefault] = []
            self._geom_attributes = []
            self._velocity_attributes = []
            pass

        def add_default(self, class_name: str):
            default = MuJoCoXMLGenerator.MuJoCoDefault(class_name)
            self._children.append(default)
            return default

        def add_geom(self, attributes):
            self._geom_attributes.append(attributes)

        def add_velocity(self, attributes):
            self._velocity_attributes.append(attributes)

        def to_string(self) -> str:
            xml = f"<default class=\"{self._class_name}\">\n"

            for a in self._geom_attributes:
                xml += "<geom "
                for k, v in a.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            for a in self._velocity_attributes:
                xml += "<velocity "
                for k, v in a.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            for d in self._children:
                xml += d.to_string()

            xml += "</default>\n"
            return xml

    class MuJoCoCustom(ToString):
        def __init__(self):
            pass

        def to_string(self) -> str:
            return "Hello"

    class MuJoCoAsset(ToString):
        def __init__(self):
            self.texture_attributes = []
            self.material_attributes = []

        def add_texture(self, attributes):
            self.texture_attributes.append(attributes)

        def add_material(self, attributes):
            self.material_attributes.append(attributes)

        def to_string(self) -> str:
            xml = "<asset>\n"
            for ta in self.texture_attributes:
                xml += "<texture "
                for k, v in ta.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"
            for ta in self.material_attributes:
                xml += "<material "
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
            self._camera = []
            self._child_body: list[MuJoCoXMLGenerator.MuJoCoBody] = []

            if self._worldbody:
                self._name = "worldbody"
            elif "name" in attributes:
                self._name = attributes["name"]

        def add_body(
                self,
                name: str = None,
                childclass: str = None,
                mocap: bool = None,
                pos: (float, float, float) = None,
                quat: (float, float, float, float) = None,
                axisangle: ((float, float, float), float) = None,
                xyaxes: ((float, float, float), (float, float, float)) = None,
                zaxis: (float, float, float) = None,
                euler: (float, float, float) = None,
                gravcomp: float = None,
                user: str = None
        ):

            """
            :param name: Name of the body.
            :param childclass: If this attribute is present, all descendant elements that admit a defaults class will use the class specified here, unless they specify their own class or another body with a child class attribute is encountered along the chain of nested bodies.Recall Default settings.
            :param mocap: If this attribute is “true”, the body is labeled as a mocap body. This is allowed only for bodies that are children of the world body and have no joints. Such bodies are fixed from the viewpoint of the dynamics, but nevertheless the forward kinematics set their position and orientation from the fields mjData.mocap_pos and mjData.mocap_quat at each time step. The size of these arrays is adjusted by the compiler so as to match the number of mocap bodies in the model. This mechanism can be used to stream motion capture data into the simulation. Mocap bodies can also be moved via mouse perturbations in the interactive visualizer, even in dynamic simulation mode. This can be useful for creating props with adjustable position and orientation. See also the mocap attribute of flag.
            :param pos: The 3D position of the body frame, in local or global coordinates as determined by the coordinate attribute of compiler. Recall the earlier discussion of local and global coordinates in Coordinate frames. In local coordinates, if the body position is left undefined it defaults to (0, 0, 0). In global coordinates, an undefined body position is inferred by the compiler through the following steps:If the inertial frame is not defined via the inertial element, it is inferred from the geoms attached to thebody. If there are no geoms, the inertial frame remains undefined. This step is applied in both local and global coordinates. If both the body frame and the inertial frame are undefined, a compile error is generated. If one of these two frames is defined and the other is not, the defined one is copied into the undefinedone. At this point both frames are defined, in global coordinates. The inertial frame as well as all elements defined in the body are converted to local coordinates, relative to the body frame. Note that whether a frame is defined or not depends on its pos attribute, which is in the special undefined state by default. Orientation cannot be used to make this determination because it has an internal default(the unit quaternion).
            :param quat: See [Frame orientations](https://mujoco.readthedocs.io/en/stable/modeling.html#corientation). Similar to position, the orientation specified here is interpreted in either local or global coordinates as determined by the coordinate attribute of compiler.Unlike position which is required in local coordinates, the orientation defaults to the unit quaternion, thus specifying it is optional even in local coordinates. If the body frame was copied from the body inertial frame per the above rules, the copy operation applies to both position and orientation, and the setting of the orientation - related attributes is ignored.
            :param axisangle: See 'quat' attribute.
            :param xyaxes: See 'quat' attribute.
            :param zaxis: See 'quat' attribute.
            :param euler: See 'quat' attribute.
            :param gravcomp: Gravity compensation force, specified as fraction of body weight. This attribute creates an upwards force applied to the body’s center of mass, countering the force of gravity. As an example, a value of 1 creates an upward force equal to the body’s weight and compensates for gravity exactly. Values greater than 1 will create a net upwards force or buoyancy effect.
            :param user: See [User parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#cuser).
            """

            attributes = {}
            for key, value in [("name", name), ("childclass", childclass), ("user", user)]:
                if value is not None:
                    attributes[key] = value

            if mocap is not None:
                attributes["mocap"] = mocap
            if pos is not None:
                attributes["pos"] = f"{pos[0]} {pos[1]} {pos[2]}"
            if gravcomp is not None:
                attributes["gravcomp"] = f"{gravcomp}"

            if axisangle is not None:
                attributes["axisangle"] = f"{axisangle[0][0]} {axisangle[0][1]} {axisangle[0][2]} {axisangle[1]}"
            elif xyaxes is not None:
                attributes["xyaxes"] \
                    = f"{xyaxes[0][0]} {xyaxes[0][1]} {xyaxes[0][2]} {xyaxes[1][0]} {xyaxes[1][1]} {xyaxes[1][2]}"
            elif zaxis is not None:
                attributes["zaxis"] \
                    = f"{zaxis[0][0]} {zaxis[0][1]} {zaxis[0][2]} {zaxis[1][0]} {zaxis[1][1]} {zaxis[1][2]}"
            elif euler is not None:
                attributes["euler"] = f"{euler[0]} {euler[1]} {euler[2]}"
            elif quat is not None:
                attributes["quat"] = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

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

        def add_camera(self, attributes):
            self._camera.append(attributes)

        def to_string(self) -> str:
            if self._worldbody:
                xml = "<worldbody>\n"
            else:
                xml = "<body "
                for k, v in self._attributes.items():
                    xml += f"{k}=\"{v}\" "
                xml += ">\n"

            for c in self._camera:
                xml += "<camera "
                for k, v in c.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

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
        class MuJoCoTendonSpatial:
            class _AttrType(enum.Enum):
                SpatialSite = 1
                SpatialGeom = 1
                SpatialPulley = 1

            def __init__(self, attributes):
                self._attributes = attributes
                self._sub_attrs = []

            def add_point(self, attributes):
                if "site" in attributes:
                    self._sub_attrs.append((self._AttrType.SpatialSite, attributes))
                elif "geom" in attributes or "sidesite" in attributes:
                    self._sub_attrs.append((self._AttrType.SpatialGeom, attributes))
                elif "divisor" in attributes:
                    self._sub_attrs.append((self._AttrType.SpatialPulley, attributes))

            def to_string(self) -> str:
                xml = "<spatial "
                for k, v in self._attributes.items():
                    xml += f"{k}=\"{v}\" "
                xml += ">\n"

                for t, attr in self._sub_attrs:
                    if t == self._AttrType.SpatialSite:
                        xml += "<site "
                    elif t == self._AttrType.SpatialGeom:
                        xml += "<geom "
                    elif t == self._AttrType.SpatialPulley:
                        xml += "<pulley "
                    for k, v in attr.items():
                        xml += f"{k}=\"{v}\" "
                    xml += "/>\n"

                xml += "</spatial>\n"

                return xml

        class MuJoCoTendonFixed:
            def __init__(self, attributes):
                self._attributes = attributes
                self._joint = []

            def add_joint(self, attributes):
                self._joint.append(attributes)

            def to_string(self) -> str:
                xml = "<fixed "
                for k, v in self._attributes.items():
                    xml += f"{k}=\"{v}\" "
                xml += ">\n"

                for j in self._joint:
                    xml += "<joint "
                    for k, v in j.items():
                        xml += f"{k}=\"{v}\" "
                    xml += "/>\n"

                xml += "</fixed>\n"

                return xml

        def __init__(self):
            self._spatial: list[MuJoCoXMLGenerator.MuJoCoTendon.MuJoCoTendonSpatial] = []
            self._fixed: list[MuJoCoXMLGenerator.MuJoCoTendon.MuJoCoTendonFixed] = []

        def add_spatial(self, attributes) -> MuJoCoTendonSpatial:
            element = MuJoCoXMLGenerator.MuJoCoTendon.MuJoCoTendonSpatial(attributes)
            self._spatial.append(element)
            return element

        def fixed(self, attributes) -> MuJoCoTendonFixed:
            element = MuJoCoXMLGenerator.MuJoCoTendon.MuJoCoTendonFixed(attributes)
            self._fixed.append(element)
            return element

        def to_string(self) -> str:
            xml = "<tendon>"
            for s in self._spatial:
                xml += s.to_string()
            for f in self._fixed:
                xml += f.to_string()
            xml += "</tendon>\n"
            return xml

    class MuJoCoActuator(ToString):
        def __init__(self):
            self.general = []
            self.motor = []
            self.position = []
            self.velocity = []

        def add_general(self, attributes):
            self.general.append(attributes)

        def add_motor(self, attributes):
            self.motor.append(attributes)

        def add_position(self, attributes):
            self.position.append(attributes)

        def add_velocity(self, attributes):
            self.velocity.append(attributes)

        def to_string(self) -> str:
            xml = "<actuator>\n"

            for a in self.general:
                xml += "<general "
                for k, v in a.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            for a in self.motor:
                xml += "<motor "
                for k, v in a.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

            for a in self.position:
                xml += "<position "
                for k, v in a.items():
                    xml += f"{k}=\"{v}\" "
                xml += "/>\n"

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

        def add_actuatorpos(self, attributes):
            element = "<actuatorpos "
            for k, v in attributes.items():
                element += f"{k}=\"{v}\" "
            element += "/>"
            self.sensor.append(element)

        def add_actuatorvel(self, attributes):
            element = "<actuatorvel "
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
        self.elements: list[MuJoCoXMLGenerator.ToString] = list([MuJoCoXMLGenerator.MuJoCoBody(True, {})])

    def add_compiler(self) -> MuJoCoCompiler:
        pass

    def add_option(self, attributes) -> MuJoCoOption:
        element = MuJoCoXMLGenerator.MuJoCoOption(attributes)
        self.elements.append(element)
        return element

    def add_size(self) -> MuJoCoSize:
        pass

    def add_visual(self) -> MuJoCoVisual:
        element = MuJoCoXMLGenerator.MuJoCoVisual()
        self.elements.append(element)
        return element

    def add_statistic(self) -> MuJoCoStatistic:
        pass

    def add_default(self) -> MuJoCoDefault:
        element = MuJoCoXMLGenerator.MuJoCoDefault("main")
        self.elements.append(element)
        return element

    def add_custom(self) -> MuJoCoCustom:
        pass

    def add_asset(self) -> MuJoCoAsset:
        element = MuJoCoXMLGenerator.MuJoCoAsset()
        self.elements.append(element)
        return element

    def get_body(self, name: str = "worldbody") -> MuJoCoBody:
        find: MuJoCoXMLGenerator.MuJoCoBody = None
        index = 0

        candidate: list[MuJoCoXMLGenerator.MuJoCoBody] = [
            e for e in self.elements if type(e) == MuJoCoXMLGenerator.MuJoCoBody
        ]

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
        element = MuJoCoXMLGenerator.MuJoCoTendon()
        self.elements.append(element)
        return element

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
