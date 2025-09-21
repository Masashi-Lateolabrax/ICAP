from flax import struct


class DiffusionVolumes:
    C = 15.9
    H = 2.31
    O = 6.11

    @staticmethod
    def ethanol() -> float:
        return DiffusionVolumes.C * 2 + DiffusionVolumes.H * 6 + DiffusionVolumes.O


# Reference(Gas Constant): https://physics.nist.gov/cgi-bin/cuu/Value?r|search_for=gas+constant
class Material:
    GAS_CONSTANT: float = 8.314462618  # J/(mol·K)=Pa·m³/(mol·K)
    AIR_MOLAR_MASS: float = 28.96  # g/mol
    AIR_DIFFUSION_VOLUME: float = 19.7

    def __init__(
            self, molar_mass: float, antoine_a: float, antoine_b: float, antoine_c: float, diffusion_volume: float
    ):
        self.molar_mass = molar_mass  # g/mol
        self.antoine_a = antoine_a
        self.antoine_b = antoine_b
        self.antoine_c = antoine_c
        self.diffusion_volume = diffusion_volume

    def saturation_pressure(self, temperature: float) -> float:
        """
        Calculate saturation pressure using the Antoine equation.

        Returns:
            float: Saturation pressure in Pa.
        """
        pressure_bar = 10 ** (self.antoine_a - (self.antoine_b / (temperature + self.antoine_c)))  # bar(=100 kPa)
        return pressure_bar * 1e5  # Pa

    def diffusion_coefficient(self, temperature: float) -> float:
        """
        Calculate diffusion coefficient using the Fuller-Schettler-Giddings equation.

        Returns:
            float: Diffusion coefficient in m^2/s.
        """

        normal_pressure = 1  # atm

        a = (10 ** -3) * (temperature ** 1.75) * (1 / self.molar_mass + 1 / self.AIR_MOLAR_MASS) ** 0.5
        b = normal_pressure * (self.diffusion_volume ** (1/3) + self.AIR_DIFFUSION_VOLUME ** (1/3)) ** 2
        diffusion = a / b  # cm^2/s
        return diffusion * 1e-4  # m^2/s


# Reference(Ethanol): https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Type=ANTOINE&Plot=on#ANTOINE
# Temperature from 292.77 to 366.63
ETHANOL = Material(
    molar_mass=46.0684,  # g/mol
    antoine_a=5.24677,
    antoine_b=1598.673,
    antoine_c=-46.424,
    diffusion_volume=DiffusionVolumes.ethanol()
)
