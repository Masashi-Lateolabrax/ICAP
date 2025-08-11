from flax import struct


@struct.dataclass
class Material:
    gas_constant: float
    molar_mass: float
    ANTOINE_A: float
    ANTOINE_B: float
    ANTOINE_C: float

    def saturation_pressure(self, temperature: float) -> float:
        """Calculate saturation pressure using the Antoine equation."""
        return 10 ** (self.ANTOINE_A - (self.ANTOINE_B / (temperature + self.ANTOINE_C))) * 1000


ETHANOL = Material(
    gas_constant=8.31446262,
    molar_mass=46.07,
    ANTOINE_A=8.04494,
    ANTOINE_B=1554.3,
    ANTOINE_C=222.65
)
