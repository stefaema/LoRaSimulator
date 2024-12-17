from enum import Enum
class LoraReservedArtifacts(Enum):
    """
    Enum class that contains the reserved artifacts of LoRa modulation.
    FULL_UPCHIRP: The full upchirp artifact. It is used to indicate the start of the package's preamble and it is the same as the symbol 0.
    FULL_DOWNCHIRP: The full downchirp artifact. It is part of the package's SFD and it is the same as the symbol 0 but with negative slope.
    QUARTER_DOWNCHIRP: The quarter downchirp artifact. It is used to indicate the end of the package's SFD and it is the same as the symbol 0 but with negative slope and only 1/4 of the duration.
    """
    FULL_UPCHIRP = (0, 1.0, 1)
    FULL_DOWNCHIRP = (0, 1.0, -1)
    QUARTER_DOWNCHIRP = (0, 0.25, -1)

    @property
    def symbol_val(self):
        return self.value[0]
    
    @property
    def duration_factor(self):
        return self.value[1]
    
    @property
    def slope_sign(self):
        return self.value[2]