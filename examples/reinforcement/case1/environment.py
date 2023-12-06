import mujoco


class EnvironmentBuilder:
    def __init__(
            self,
            pos,
            pole_length: float,
            kv: float,
            body_weight=1000,
            pole_weight=1000,
            timestep: float = 0.002,
            name: str = "Title",
    ):
        """
        :param pos: The position of vehicle. Its type should be like tuple[float,float,float].
        :param weight: The weight of stone-weight.
        :param kv: The feedback gain of velocity actuator.
        :param timestep: The timestep is used to simulate.
        :param name: Name of the body.
        """

        self.name = name
        self.timestep = timestep
        self.pos = pos
        self.pole_length = pole_length
        self.weight = {
            "pole": pole_weight,
            "body": body_weight,
        }
        self.kv = kv

    def build(self) -> mujoco.MjModel:
        xml = fr"""
        <mujoco model="{self.name}">
        
        <option timestep="{self.timestep}">
        </option>
    
        <default>
        </default>
        
        <asset>
        </asset>
        
        <visual>
            <headlight ambient="0.5 0.5 0.5" diffuse="0.2 0.2 0.2" specular="0.7 0.7 0.7"/>
        </visual>
    
        <worldbody>
            <geom name="ground" type="plane" size="0 0 1" rgba="1.0 1.0 1.0 0.3"/>
                
            <body pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}">
                <joint type="slide" axis="0 0 1" />
                <joint name="vehicle_joint" type="slide" axis="1 0 0"/>
                
                <site name="body_site"/>
                
                <geom name="vehicle_body1" type="box" size="2.5 0.5 0.5" pos="0 1.0 0" rgba="1.0 1.0 1.0 1.0" condim="3" priority="1"  mass="{self.weight["body"]}"/>
                <geom name="vehicle_body2" type="box" size="2.5 0.5 0.5" pos="0 -1.0 0" rgba="1.0 1.0 1.0 1.0" condim="3" priority="1" mass="{self.weight["body"]}"/>
                
                <body pos="0 0 -{self.pole_length * 0.5}">
                    <joint name="pole_joint" type="hinge" axis="0 1 0" pos="0 0 {self.pole_length * 0.5}"/>
                    
                    <geom name="pole" type="cylinder" size="0.5 {self.pole_length * 0.5}" mass="{self.weight["pole"]}" pos="0 0 0" rgba="1.0 1.0 1.0 1.0" contype="2" conaffinity="2"/>
                </body>
            </body>
        </worldbody>
        
        <actuator>
            <velocity name="vehicle_act" joint="vehicle_joint" kv="{self.kv}"/>
        </actuator>
        
        <sensor>
            <velocimeter name="body_velocity" site="body_site"/>
            <force name="body_force" site="body_site"/>
            
            <jointpos name="pole_angle" joint="pole_joint"/>
            <jointvel name="pole_velocity" joint="pole_joint"/>
        </sensor>
        
        </mujoco>
        """

        # noinspection PyArgumentList
        return mujoco.MjModel.from_xml_string(xml=xml)
