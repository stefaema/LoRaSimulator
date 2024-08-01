from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

class LoraReservedArtifacts(Enum):
    FULL_UPCHIRP = 0
    FULL_DOWNCHIRP = 1
    QUARTER_DOWNCHIRP = 2
    
class LoraModulator():
    def __init__(self, spreading_factor, bandwidth, samples_per_chirp = 1):
        # LoRa technique external parameters
        self.__spreading_factor = spreading_factor
        self.__bandwidth = bandwidth
        self.__samples_per_chirp = samples_per_chirp
        # Lora technique internal parameters
        self.__chips_number = 2 ** spreading_factor
        self.__symbol_duration = ( self.__chips_number - 1 ) / bandwidth
        self.__samples_per_symbol = int(self.__chips_number * samples_per_chirp)
        self.__frequency_slope = (bandwidth ** 2) / (self.__chips_number - 1)
        self.__sampling_period = self.__symbol_duration / self.__samples_per_symbol
    
    def generate_frequency_evolution_across_time(self, symbols, return_only_offset_time_axis = True):

        frequency_evolution = []
        time_axis = []
        time_axis_without_offset = []
        current_offset = 0
        
        for symbol in symbols:

            current_slope = self.__frequency_slope
            symbol_time_axis = np.linspace(current_offset, current_offset + self.__symbol_duration, self.__samples_per_symbol)

            if symbol == LoraReservedArtifacts.FULL_UPCHIRP:
                symbol = 0

            elif symbol == LoraReservedArtifacts.FULL_DOWNCHIRP:
                symbol = 2**self.__spreading_factor
                current_slope = - current_slope
            
            elif symbol == LoraReservedArtifacts.QUARTER_DOWNCHIRP:
                symbol = 2**self.__spreading_factor
                current_slope = - current_slope
                quarter_cycle_max_index = int(len(symbol_time_axis) // 4 + 1)
                symbol_time_axis = symbol_time_axis[:quarter_cycle_max_index]

            y_intercept = symbol * ( self.__bandwidth / (2**self.__spreading_factor) )
            
            for i in range(len(symbol_time_axis)):
                instantaneous_frequency = y_intercept + current_slope * (symbol_time_axis[i] - current_offset)

                if instantaneous_frequency > self.__bandwidth:
                    # Not necesarry to take into account multiples of bandwidth, as the duration of the chirp is limited 
                    instantaneous_frequency -= self.__bandwidth


                frequency_evolution.append(instantaneous_frequency)

            time_axis.extend(symbol_time_axis)
            symbol_time_axis_without_offset = symbol_time_axis - current_offset
            time_axis_without_offset.extend(symbol_time_axis_without_offset)
            
            time_stop = symbol_time_axis[-1] + self.__sampling_period
            current_offset = time_stop
            
            

            if return_only_offset_time_axis:
                return time_axis, frequency_evolution
            if not return_only_offset_time_axis:
                return time_axis, time_axis_without_offset,frequency_evolution
            return time_axis, frequency_evolution
    
    def generate_chirp_from_frequency_evolution(self, time_axis, frequency_evolution):
        coefficient = 1/np.sqrt(2**self.__spreading_factor)
        signal = []
        for i in range(len(time_axis)):
            instantaneous_phase = 1j * 2 * np.pi * frequency_evolution[i] * time_axis[i]
            signal_sample = coefficient * np.exp(instantaneous_phase)
            signal.append(signal_sample)
        return signal

    def modulate_symbols(self, symbols, also_return_frequency_evolution = False):
        for symbol in symbols:
            if symbol not in range(self.__chips_number) and not isinstance(symbol, LoraReservedArtifacts):
                raise ValueError(f"Symbols have to be of type LoraReservedArtifacts or integers between 0 and {self.__chips_number - 1} for the given spreading factor")
        time_axis, no_offset_time_axis, frequency_evolution = self.generate_frequency_evolution_across_time(symbols, False)
        signal = self.generate_chirp_from_frequency_evolution(no_offset_time_axis, frequency_evolution)
        if not also_return_frequency_evolution:
            return time_axis, signal
        return time_axis, signal, frequency_evolution

    
    def modulate_implicit_package(self,preamble_number, payload, also_return_frequency_evolution = False):
        package = []
        for i in range(preamble_number):
            package.append(LoraReservedArtifacts.FULL_UPCHIRP)
        for i in range(2):
            package.append(LoraReservedArtifacts.FULL_UPCHIRP)
        for i in range(2):
            package.append(LoraReservedArtifacts.FULL_DOWNCHIRP)
        package.append(LoraReservedArtifacts.QUARTER_DOWNCHIRP)

        package.extend(payload)
        time_axis, signal, frequency_evolution = self.modulate_symbols(package, True)
        if not also_return_frequency_evolution:
            return time_axis, signal
        return time_axis, signal, frequency_evolution
    
    @property
    def spreading_factor(self):
        return self.__spreading_factor
    @spreading_factor.setter
    def spreading_factor(self, value):
        if value not in [7, 8, 9, 10, 11, 12]:
            raise ValueError("Spreading factor has to be one of the integers: 7, 8, 9, 10, 11 or 12")
        self.__spreading_factor = value
        self.__chips_number = 2 ** value
        self.__symbol_duration = self.__chips_number / self.__bandwidth
        self.__samples_per_symbol = int(self.__chips_number * self.resolution_between_chips)
        self.__sampling_period = self.__symbol_duration / self.__samples_per_symbol
        self.__frequency_slope = (self.__bandwidth ** 2) / self.__chips_number

    @property
    def bandwidth(self):
        return self.__bandwidth
    @bandwidth.setter
    def bandwidth(self, value):
        if value not in [125, 250, 500]:
            raise ValueError("Bandwidth has to be one of the integers: 125, 250 or 500. Remember that it is in kHz.")
        self.__bandwidth = value
        self.__symbol_duration = self.__chips_number / value
        self.__samples_per_symbol = int(self.__chips_number * self.resolution_between_chips)
        self.__sampling_period = self.__symbol_duration / self.__samples_per_symbol
        self.__frequency_slope = (value ** 2) / self.__chips_number

    @property
    def resolution_between_chips(self):
        return self.__resolution_between_chips
    @resolution_between_chips.setter
    def resolution_between_chips(self, value):
        if value < 1:
            raise ValueError("Resolution between chips has to be greater than 0")
        if value % 2 != 0:
            print("Be careful, the resolution between chips is not a multiple of 2. This may lead to errors.")
        self.__resolution_between_chips = value
        self.__samples_per_symbol = int(self.__chips_number * value)
        self.__sampling_period = self.__symbol_duration / self.__samples_per_symbol


class LoraDemodulator:
    def __init__(self, spreading_factor, bandwidth,oversampling_factor = 1):
        # The spreading factor of the LoRa modulation
        self.spreading_factor = spreading_factor
        
        # The bandwidth of the LoRa modulation in KiloHertz
        self.bandwidth = bandwidth

        self.__oversampling_factor = oversampling_factor
    
    def generate_full_downchirp(self):
        # Number of discrete levels in the transmitted symbol that eventually will be mapped to a frequency
        chips_number = 2**self.spreading_factor

        # Number of samples per symbol
        samples_per_symbol = int(chips_number)

        # Time duration of each symbol
        symbol_duration = (chips_number - 1)/np.abs(self.bandwidth)

        sampling_period = symbol_duration/samples_per_symbol

        # Slope of the downchirp, defined by the bandwidth and the time duration of each symbol
        slope = - (self.bandwidth**2) /(chips_number - 1)

        y_intercept = self.bandwidth

        # Frequency evolution of the chirp signal
        symbol_frequency_evolution = []

        symbol_time_axis = np.linspace(0, symbol_duration, samples_per_symbol * self.__oversampling_factor)
        # Signal
        signal = []

        for i in range(len(symbol_time_axis)):
                
            # Instantaneous frequency of the chirp signal
            instantaneous_frequency = ( y_intercept + slope * symbol_time_axis[i] )

            symbol_frequency_evolution.append(instantaneous_frequency) 

            instantaneous_phase = 1j * 2 * np.pi * instantaneous_frequency * symbol_time_axis[i] 
            coefficient = 1 / (np.sqrt(2**self.spreading_factor))
            signal.append(coefficient * np.exp(instantaneous_phase))

        return symbol_time_axis, symbol_frequency_evolution, signal
        
    def dechirp_symbol(self, signal):
        downchirp_signal = self.generate_full_downchirp()[2]
        dechirped_signal = [signal[i] * downchirp_signal[i] for i in range(len(signal))]
        return dechirped_signal
    
    def demodulate_symbol(self, signal, return_fft = False):
        dechirped_signal = self.dechirp_symbol(signal)
        correlation = np.fft.fft(dechirped_signal)
        observed_symbol = np.argmax(correlation)
        if return_fft:
            return correlation
        return observed_symbol