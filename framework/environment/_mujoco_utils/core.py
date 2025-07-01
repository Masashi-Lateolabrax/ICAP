import mujoco
import numpy as np


def add_texture(
        spec: mujoco.MjSpec,
        name: str,
        type_: mujoco.mjtTexture,
        builtin: mujoco.mjtBuiltin,
        width: int,
        height: int,
        rgb1: tuple[float, float, float],
        rgb2: tuple[float, float, float]
) -> mujoco._specs.MjsTexture:
    """Add a texture to the MuJoCo simulation specification.
    
    Args:
        spec: MuJoCo simulation specification
        name: Name identifier for the texture
        type_: Type of texture (e.g., 2D texture)
        builtin: Built-in texture pattern type
        width: Texture width in pixels
        height: Texture height in pixels
        rgb1: Primary RGB color values (0.0-1.0)
        rgb2: Secondary RGB color values (0.0-1.0)
        
    Returns:
        Created MuJoCo texture specification object
    """
    texture = spec.add_texture(
        name=name,
        type=type_,
        builtin=builtin,
        width=width,
        height=height,
        rgb1=rgb1,
        rgb2=rgb2,
    )
    return texture


def add_material(
        spec: mujoco.MjSpec,
        name: str,
        texture: str,
        texrepeat: tuple[float, float],
) -> mujoco._specs.MjsMaterial:
    """Add a material to the MuJoCo simulation specification.
    
    Args:
        spec: MuJoCo simulation specification
        name: Name identifier for the material
        texture: List of texture names to apply to the material
        texrepeat: Texture repetition factors (x, y)
        
    Returns:
        Created MuJoCo material specification object
    """
    material: mujoco._specs.MjsMaterial = spec.add_material(name=name, texrepeat=texrepeat)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture
    return material


def add_geom(
        body: mujoco._specs.MjsBody,
        geom_type: mujoco.mjtGeom,
        size: tuple[float, float, float],
        pos: tuple[float, float, float] = (0, 0, 0),
        name: str = None,
        material: str = None,
        rgba: tuple[float, float, float, float] = None,
        condim: int = None,
        density: float = None,
        mass: float = None,
        quat: np.ndarray = None,
) -> mujoco._specs.MjsGeom:
    """Add a geometry to a MuJoCo body.
    
    Args:
        body: Parent body to attach the geometry to
        geom_type: Type of geometry (box, cylinder, sphere, etc.)
        size: Dimensions of the geometry (x, y, z)
        pos: Position relative to parent body (x, y, z)
        name: Optional name identifier for the geometry
        material: Optional material name to apply
        rgba: Optional color and transparency (r, g, b, a)
        condim: Optional contact dimensionality for collision detection
        density: Optional density for mass calculation
        mass: Optional explicit mass (overrides density)
        
    Returns:
        Created MuJoCo geometry specification object
        
    Note:
        If both mass and density are provided, mass takes precedence.
    """
    geom: mujoco._specs.MjsGeom = body.add_geom()
    geom.type = geom_type
    geom.pos = pos
    geom.size = size

    if quat is not None:
        geom.quat = quat
    if name is not None:
        geom.name = name
    if material is not None:
        geom.material = material
    if rgba is not None:
        geom.rgba = rgba
    if condim is not None:
        geom.condim = condim
    if mass is not None:
        geom.mass = mass
    elif density is not None:
        geom.density = density

    return geom


def add_site(
        body: mujoco._specs.MjsBody,
        name: str,
        pos: tuple[float, float, float] = None,
        size: list[float] = None,
        rgba: tuple[float, float, float, float] = None,
        type_: mujoco.mjtGeom = None,
) -> mujoco._specs.MjsSite:
    """Add a site (reference point) to a MuJoCo body.
    
    Sites are used for sensors, visualization, and as reference points
    for measurements and attachments.
    
    Args:
        body: Parent body to attach the site to
        name: Name identifier for the site
        pos: Optional position relative to parent body (x, y, z)
        size: Optional size parameters for site visualization
        rgba: Optional color and transparency (r, g, b, a)
        type_: Optional geometry type for site visualization
        
    Returns:
        Created MuJoCo site specification object
    """
    site: mujoco._specs.MjsSite = body.add_site()
    site.name = name

    if pos is not None:
        site.pos = pos
    if size is not None:
        site.size = size
    if rgba is not None:
        site.rgba = rgba
    if type_ is not None:
        site.type = type_

    return site


def add_body(
        parent: mujoco._specs.MjsBody,
        pos: tuple[float, float, float] = (0, 0, 0),
        name: str = None,
        quat: tuple[float, float, float, float] = None,
) -> mujoco._specs.MjsBody:
    """Add a child body to a parent body in the MuJoCo simulation.
    
    Args:
        parent: Parent body to attach the new body to
        pos: Position relative to parent body (x, y, z)
        name: Optional name identifier for the body
        quat: Optional quaternion orientation (w, x, y, z)
        
    Returns:
        Created MuJoCo body specification object
    """
    body: mujoco._specs.MjsBody = parent.add_body()
    body.pos = pos

    if name is not None:
        body.name = name
    if quat is not None:
        body.quat = quat

    return body


def add_joint(
        body: mujoco._specs.MjsBody,
        name: str,
        joint_type: mujoco.mjtJoint,
        axis: tuple[float, float, float] = None,
        pos: tuple[float, float, float] = None,
        limited: bool = None,
        range_: tuple[float, float] = None,
        stiffness: float = None,
        damping: float = None,
) -> mujoco._specs.MjsJoint:
    """Add a joint to a MuJoCo body.
    
    Args:
        body: Parent body to attach the joint to
        name: Name identifier for the joint
        joint_type: Type of joint (hinge, slide, free, etc.)
        axis: Optional axis of rotation/translation (x, y, z)
        pos: Optional position relative to parent body (x, y, z)
        limited: Optional flag to enable joint limits
        range_: Optional joint range limits (min, max)
        stiffness: Optional spring stiffness coefficient
        damping: Optional damping coefficient
        
    Returns:
        Created MuJoCo joint specification object
    """
    joint: mujoco._specs.MjsJoint = body.add_joint()
    joint.name = name
    joint.type = joint_type

    if axis is not None:
        joint.axis = axis
    if pos is not None:
        joint.pos = pos
    if limited is not None:
        joint.limited = limited
    if range_ is not None:
        joint.range = range_
    if stiffness is not None:
        joint.stiffness = stiffness
    if damping is not None:
        joint.damping = damping

    return joint


def _create_base_general_actuator(
        spec: mujoco.MjSpec,
        dyntype: mujoco.mjtDyn,
        dynprm: tuple[float, float, float],
        gaintype: mujoco.mjtGain,
        gainprm: tuple[float, float, float],
        biastype: mujoco.mjtBias,
        biasprm: tuple[float, float, float],
        name: str = None,
        gear: tuple[float, float, float, float, float, float] = None
) -> mujoco._specs.MjsActuator:
    """Create a base general actuator with specified dynamics, gain, and bias.
    
    This is a helper function for creating specialized actuators with
    common parameter configurations.
    
    Args:
        spec: MuJoCo simulation specification
        dyntype: Dynamics type for actuator behavior
        dynprm: Dynamics parameters (up to 3 values)
        gaintype: Gain type for actuator control
        gainprm: Gain parameters (up to 3 values)
        biastype: Bias type for actuator offset
        biasprm: Bias parameters (up to 3 values)
        name: Optional name identifier for the actuator
        
    Returns:
        Created MuJoCo actuator specification object
    """
    actuator: mujoco._specs.MjsActuator = spec.add_actuator()

    actuator.dyntype = dyntype
    actuator.dynprm[0:3] = dynprm

    actuator.gaintype = gaintype
    actuator.gainprm[0:3] = gainprm

    actuator.biastype = biastype
    actuator.biasprm[0:3] = biasprm

    if name is not None:
        actuator.name = name
    if gear is not None:
        actuator.gear = gear

    return actuator


def add_velocity_actuator(
        spec: mujoco.MjSpec,
        joint: mujoco._specs.MjsJoint,
        kv: float,
        name: str = None,
        gear: tuple[float, float, float, float, float, float, float] = None
) -> mujoco._specs.MjsActuator:
    """Add a velocity actuator to control a joint.
    
    Creates a velocity-controlled actuator that applies forces/torques
    proportional to the velocity error.
    
    Args:
        spec: MuJoCo simulation specification
        joint: Target joint to control
        kv: Velocity gain coefficient
        name: Optional name identifier for the actuator
        
    Returns:
        Created MuJoCo actuator specification object
        
    Raises:
        ValueError: If spec, joint is None, or kv is invalid
        TypeError: If kv is not a number
    """
    if spec is None:
        raise ValueError("MuJoCo specification cannot be None")
    if joint is None:
        raise ValueError("Joint cannot be None")
    if not isinstance(kv, (int, float)):
        raise TypeError("Velocity gain (kv) must be a number")
    if kv <= 0:
        raise ValueError("Velocity gain (kv) must be positive")
    actuator = _create_base_general_actuator(
        spec,
        dyntype=mujoco.mjtDyn.mjDYN_NONE,
        dynprm=(1, 0, 0),
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=(kv, 0, 0),
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=(0, 0, -kv),
        name=name,
        gear=gear
    )
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = joint.name
    return actuator


def add_position_actuator(
        spec: mujoco.MjSpec,
        joint: mujoco._specs.MjsJoint,
        kp: float,
        kv: float,
        name: str = None,
        gear: tuple[float, float, float, float, float, float, float] = None
) -> mujoco._specs.MjsActuator:
    if spec is None:
        raise ValueError("MuJoCo specification cannot be None")
    if joint is None:
        raise ValueError("Joint cannot be None")
    actuator = _create_base_general_actuator(
        spec,
        dyntype=mujoco.mjtDyn.mjDYN_NONE,
        dynprm=(0, 0, 0),
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=(kp, 0, 0),
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=(0, -kp, -kv),
        name=name,
        gear=gear
    )
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = joint.name
    return actuator


def add_sensor(
        spec: mujoco.MjSpec,
        sensor_type: mujoco.mjtSensor,
        name: str = None,
        noise: float = None,
        cutoff: float = None,
        **kwargs
) -> mujoco._specs.MjsSensor:
    """Add a sensor to the MuJoCo simulation specification.
    
    Args:
        spec: MuJoCo simulation specification
        sensor_type: Type of sensor (velocimeter, accelerometer, etc.)
        name: Optional name identifier for the sensor
        noise: Optional noise level for sensor readings
        cutoff: Optional cutoff frequency for sensor filtering
        **kwargs: Additional sensor-specific parameters
        
    Returns:
        Created MuJoCo sensor specification object
    """
    sensor: mujoco._specs.MjsSensor = spec.add_sensor()

    sensor.type = sensor_type

    if name is not None:
        sensor.name = name
    if noise is not None:
        sensor.noise = noise
    if cutoff is not None:
        sensor.cutoff = cutoff

    for key, value in filter(lambda x: x[1] is not None, kwargs.items()):
        setattr(sensor, key, value)
    return sensor


def add_velocimeter(
        spec: mujoco.MjSpec,
        name: str,
        site: mujoco._specs.MjsSite,
        noise: float = None,
        cutoff: float = None,
) -> mujoco._specs.MjsSensor:
    """Add a velocimeter sensor to measure velocity at a site.
    
    Args:
        spec: MuJoCo simulation specification
        name: Name identifier for the velocimeter
        site: Site where velocity is measured
        noise: Optional noise level for velocity readings
        cutoff: Optional cutoff frequency for velocity filtering
    """
    return add_sensor(
        spec,
        name=name,
        noise=noise,
        cutoff=cutoff,
        sensor_type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=site.name,
    )
