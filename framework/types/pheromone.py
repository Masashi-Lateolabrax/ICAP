from flax import struct


@struct.dataclass
class Material:
    gas_constant: float
    molar_mass: float
    ANTOINE_A: float
    ANTOINE_B: float
    ANTOINE_C: float

    def saturation_pressure(self, temperature: float) -> float:
        """
        Calculate saturation pressure using the Antoine equation.

        Returns:
            float: Saturation pressure in kPa.
        """
        return 10 ** (self.ANTOINE_A - (self.ANTOINE_B / (temperature + self.ANTOINE_C))) * 100


# Reference(Gas Constant): https://physics.nist.gov/cgi-bin/cuu/Value?r|search_for=gas+constant
# Reference(Ethanol): https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Type=ANTOINE&Plot=on#ANTOINE
# Temperature from 292.77 to 366.63
ETHANOL = Material(
    gas_constant=8.314462618, # J/(mol·K)
    molar_mass=46.0684, # g/mol
    ANTOINE_A=5.24677,
    ANTOINE_B=1598.673,
    ANTOINE_C=-46.424
)
