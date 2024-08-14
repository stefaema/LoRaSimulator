from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

class LoraReservedArtifacts(Enum):
    FULL_UPCHIRP = -1
    FULL_DOWNCHIRP = -2
    QUARTER_DOWNCHIRP = -3

class LoraModulator():
    def validate_parameters(self):
        '''
        This function validates the parameters of the LoRa modulation.
        '''
        if self._spreading_factor not in range(7, 13) or not isinstance(self._spreading_factor, int):
            raise ValueError('The spreading factor must be an integer between 7 and 12.')
        if  not isinstance(self._bandwidth, (int, float)):
            raise ValueError('The bandwidth must be an integer or float. Tipically, it is 125e3 Hz, 250e3 Hz or 500e3 Hz.')
        if self._bandwidth not in [125e3, 250e3, 500e3, 125000, 25000, 500000]:
            print('WARNING: Bandwidth typically takes values of 125e3 Hz, 250e3 Hz, or 500e3 Hz.')
        if self._samples_per_chip < 1 or not isinstance(self._samples_per_chip, int):
            raise ValueError('The number of samples per chip must be an integer greater than 0.')
        
    def __init__(self, sf, bw, samples_per_chip = 1):
        self._spreading_factor = sf
        self._bandwidth = bw
        self._samples_per_chip = samples_per_chip
        self.validate_parameters()

        self._chips_number = 2**sf
        self._symbol_duration = 2**sf / bw
        self._frequency_slope = bw / self._symbol_duration
        self._signal_coefficient = 1 / np.sqrt(2**self._spreading_factor * self._samples_per_chip)

    def generate_timeline(self, symbols):
        '''
        This function generates the timeline of the LoRa modulation.

        Parameters:
        
        num_symbols (int): The number of symbols in the modulation.

        Returns:

        timeline (np.array): The timeline of the LoRa modulation.
        '''
        num_symbols = len(symbols)
        if LoraReservedArtifacts.QUARTER_DOWNCHIRP in symbols:
            num_symbols = num_symbols - 0.75

        modulation_duration = num_symbols * self._symbol_duration

        samples_per_symbol = self._samples_per_chip * self._chips_number
        samples_per_modulation = int(samples_per_symbol * num_symbols)
        timeline = np.linspace(0, modulation_duration, samples_per_modulation, endpoint = False)
        return timeline

    def generate_instantaneous_frequency(self, symbols):
        '''
        This function generates the instaneous frequency of the LoRa modulation.

        Parameters:

        timeline (np.array): The timeline of the LoRa modulation.

        Returns:

        instaneous_frequency (np.array): The instaneous frequency of the LoRa modulation.
        '''
        whole_frequency_evolution = []
        spc = self._samples_per_chip
        sf = self._spreading_factor

        for symbol in symbols:
            aux_factor = 1
            sample_range = range(spc * 2**sf)
            if symbol not in range(0, 2**sf) and not isinstance(symbol, LoraReservedArtifacts):
                raise ValueError('The symbol must be an integer between 0 and 2^SF - 1 or a LoRa Reserved Artifact.')
            if isinstance(symbol, LoraReservedArtifacts):
                if symbol == LoraReservedArtifacts.FULL_UPCHIRP:
                    symbol = 0
                elif symbol == LoraReservedArtifacts.FULL_DOWNCHIRP:
                    symbol = 0
                    aux_factor = -1

                elif symbol == LoraReservedArtifacts.QUARTER_DOWNCHIRP:
                    symbol = 0
                    aux_factor = -1
                    sample_range = range(spc * 2**sf // 4)
                    
            y_intercept = symbol * (self._bandwidth / 2**sf)
            symbol_frequency_evolution = [ ( y_intercept + aux_factor * k/(self._symbol_duration * spc) ) % (aux_factor * self._bandwidth) for k in sample_range]
            whole_frequency_evolution.extend(symbol_frequency_evolution)
        
        return whole_frequency_evolution

    def generate_instantaneous_phase(self, symbols):
        '''
        This function generates the instaneous phase of the LoRa modulation.

        Parameters:

        symbols (list): The list of symbols in the modulation.

        Returns:

        instantaneous_phase (np.array): The instaneous phase of the LoRa modulation.
        '''
        spc = self._samples_per_chip
        sf = self._spreading_factor
        whole_phase_evolution = []
        
        for symbol in symbols:
            aux_factor = 1
            samples_range = range(spc * 2**sf)
            if symbol not in range(0, 2**sf) and not isinstance(symbol, LoraReservedArtifacts):
                raise ValueError('The symbol must be an integer between 0 and 2^SF - 1 or a LoRa Reserved Artifact.')
            if isinstance(symbol, LoraReservedArtifacts):
                if symbol == LoraReservedArtifacts.FULL_UPCHIRP:
                    symbol = 0
                elif symbol == LoraReservedArtifacts.FULL_DOWNCHIRP:
                    symbol = 0
                    aux_factor = -1
                elif symbol == LoraReservedArtifacts.QUARTER_DOWNCHIRP:
                    symbol = 0
                    aux_factor = -1
                    samples_range = range(spc * 2**sf // 4)

            symbol_phase_evolution_1 = [2 * np.pi *(symbol + aux_factor * k/(2*spc))*(k/(2**sf * spc)) for k in samples_range]
            indicator_function = [1 if ( (k/spc) > (2**sf - symbol) ) else 0 for k in samples_range]
            symbol_phase_evolution_2 = [( -2*np.pi * (k/spc - (2**sf - symbol)) ) * indicator_function[k] for k in samples_range]
            symbol_phase_evolution = [symbol_phase_evolution_1[k] + symbol_phase_evolution_2[k] for k in samples_range]
            whole_phase_evolution.extend(symbol_phase_evolution)
        return whole_phase_evolution

    def generate_signal(self, symbols):
        '''
        This function generates the signal of the LoRa modulation.

        Parameters:

        symbols (list): The list of symbols in the modulation.

        Returns:

        signal (np.array): The signal of the LoRa modulation.
        '''
        instantaneous_phase = self.generate_instantaneous_phase(symbols)
        signal = [self._signal_coefficient * np.exp(1j * phase) for phase in instantaneous_phase]
        return signal
    
    def modulate_symbols(self, symbols):
        '''
        This function modulates the symbols in the LoRa modulation.

        Parameters:

        symbols (list): The list of symbols in the modulation.

        Returns:

        signal (np.array): The signal of the LoRa modulation.
        '''
        timeline = self.generate_timeline(symbols)
        frequency_evolution = self.generate_instantaneous_frequency(symbols)
        signal = self.generate_signal(symbols)
        return timeline, frequency_evolution, signal
    
    def modulate_implicit_package(self, preamble_number, payload):
        '''
        This function modulates an implicit package in the LoRa modulation.

        Parameters:

        preamble_number (int): The number of preambles in the package.
        payload (list): The payload of the package.

        Returns:

        signal (np.array): The signal of the LoRa modulation.
        '''
        package = []
        for i in range(preamble_number):
            package.append(LoraReservedArtifacts.FULL_UPCHIRP)
        for i in range(2):
            package.append(LoraReservedArtifacts.FULL_UPCHIRP)
        for i in range(2):
            package.append(LoraReservedArtifacts.FULL_DOWNCHIRP)
        package.append(LoraReservedArtifacts.QUARTER_DOWNCHIRP)

        package.append(len(payload))
        package.extend(payload)
        time_axis, frequency_evolution, signal = self.modulate_symbols(package)
        return time_axis, frequency_evolution, signal

        
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

class LoraDemodulator():
    def validate_parameters(self):
        '''
        This function validates the parameters of the LoRa modulation.
        '''
        if self._spreading_factor not in range(7, 13) or not isinstance(self._spreading_factor, int):
            raise ValueError('The spreading factor must be an integer between 7 and 12.')
        if  not isinstance(self._bandwidth, (int, float)):
            raise ValueError('The bandwidth must be an integer or float. Tipically, it is 125e3 Hz, 250e3 Hz or 500e3 Hz.')
        if self._bandwidth not in [125e3, 250e3, 500e3, 125000, 25000, 500000]:
            print('WARNING: Bandwidth typically takes values of 125e3 Hz, 250e3 Hz, or 500e3 Hz.')
        if self._samples_per_chip < 1 or not isinstance(self._samples_per_chip, int):
            raise ValueError('The number of samples per chip must be an integer greater than 0.')
        
    def __init__(self, sf, bw, samples_per_chip = 1):
        self._spreading_factor = sf
        self._bandwidth = bw
        self._samples_per_chip = samples_per_chip
        self.validate_parameters()

        self._chips_number = 2**sf
        self._symbol_duration = 2**sf / bw
        self._frequency_slope = bw / self._symbol_duration
        self._signal_coefficient = 1 / np.sqrt(2**self._spreading_factor * self._samples_per_chip)

        self._base_signals = self.generate_base_signals()

    def generate_downchirp(self):
        '''
        This function generates the downchirp of the LoRa modulation.

        Returns:

        downchirp (np.array): The downchirp of the LoRa modulation.
        '''
        spc = self._samples_per_chip
        sf = self._spreading_factor
        samples_range = range(spc * 2**sf)
        instantanous_phase = [2 * np.pi * (-k/(2*spc)) * (k/(2**sf * spc)) for k in samples_range]
        downchirp_signal = [self._signal_coefficient * np.exp(1j * phase) for phase in instantanous_phase]
        return downchirp_signal
    
    def generate_upchirp(self):
        '''
        This function generates the upchirp of the LoRa modulation.

        Returns:

        upchirp (np.array): The upchirp of the LoRa modulation.
        '''
        spc = self._samples_per_chip
        sf = self._spreading_factor
        samples_range = range(spc * 2**sf)
        instantanous_phase = [2 * np.pi * (k/(2*spc)) * (k/(2**sf * spc)) for k in samples_range]
        upchirp_signal = [self._signal_coefficient * np.exp(1j * phase) for phase in instantanous_phase]
        return upchirp_signal

    def generate_quarter_upchirp(self):
        '''
        This function generates the quarter upchirp of the LoRa modulation.

        Returns:

        quarter_upchirp (np.array): The quarter upchirp of the LoRa modulation.
        '''
        spc = self._samples_per_chip
        sf = self._spreading_factor
        samples_range = range((spc * 2**sf)//4)
        instantanous_phase = [2 * np.pi * (k/(2*spc)) * (k/(2**sf * spc)) for k in samples_range]
        quarter_upchirp_signal = [self._signal_coefficient * np.exp(1j * phase) for phase in instantanous_phase]
        return quarter_upchirp_signal
    
    def generate_base_signals(self):
        upchirp = self.generate_upchirp()
        downchirp = self.generate_downchirp()
        quarter_upchirp = self.generate_quarter_upchirp()
        base_signals = {
            'upchirp': upchirp,
            'downchirp': downchirp,
            'quarter_upchirp': quarter_upchirp
        }
        return base_signals
    
    def adjust_correlation(self, correlation):
        spc = self._samples_per_chip
        sf = self._spreading_factor
        if spc ==1:
            return correlation
        
        fft_ideal_length = 2**self._spreading_factor
        new_correlation = [correlation[i]+correlation[i + (spc-1)*2**sf] for i in range(fft_ideal_length)]
        return new_correlation


    def demodulate_symbol(self, symbol_signal, base_fn = 'downchirp'):
        '''
        This function demodulates a symbol in the LoRa modulation.

        Parameters:

        symbol_signal (np.array): The signal of the symbol in the modulation.
        base_fn (np.array): A string indicating the base function of the symbol in the modulation that shall be used.

        Returns:

        symbol (int): The symbol of the LoRa modulation.
        '''
        if base_fn not in ['downchirp', 'upchirp', 'quarter_upchirp']:
            raise ValueError('The base function must be either "downchirp", "upchirp" or "quarter_upchirp".')
        base_signal = self._base_signals.get(base_fn)
        if base_signal is None:
            raise ValueError(f'The base function {base_fn} is not recognized. It must be either "downchirp", "upchirp" or "quarter_upchirp".')
        dechirped_signal = [symbol_signal[i] * base_signal[i] for i in range(len(symbol_signal))]
        correlation = np.fft.fft(dechirped_signal)
        # correlation = self.adjust_correlation(correlation)
        symbol = np.argmax(correlation) % 2**self._spreading_factor
        return symbol

    def demodulate_symbols(self, signal, base_fn = 'downchirp'):
        '''
        This function demodulates the symbols in the LoRa modulation payload, given that the received signal is already synchronized.

        Parameters:

        signal (np.array): The signal of the received LoRa modulation.
        base_fn (np.array): A string indicating the base function of the symbol in the modulation that shall be used to correlate the signal to its embedded symbol.

        Returns:

        symbols (list): The list of symbols in the LoRa modulation.
        '''
        spc = self._samples_per_chip
        sf = self._spreading_factor
        symbols = []
        for i in range(0, len(signal), spc * 2**sf):
            symbol_signal = signal[i:i + spc * 2**sf]
            symbol = self.demodulate_symbol(symbol_signal, base_fn)
            symbols.append(symbol)
        return symbols

    def compute_synchronized_index(self, signal, preamble_number):
        '''
        This function computes the synchronized index of the LoRa modulation.

        Parameters:

        signal (np.array): The signal of the received LoRa modulation.
        preamble_number (int): The number of preambles in the package.

        Returns:

        synchronized_index (int): The synchronized index of the LoRa modulation.
        '''
        
        spc = self._samples_per_chip
        sf = self._spreading_factor
        
        correct_demodulated_preamble = [0 for i in range(preamble_number + 2)]
        for i in range(0, len(signal)):
            first_preamble_symbol_signal = signal[i:i + spc * 2**sf]
            symbol = self.demodulate_symbol(first_preamble_symbol_signal)
            if symbol == 0:
                print('An upchirp was encountered at ', i)
                preamble_symbol_signal = signal[i:(i + spc * 2**sf * (preamble_number + 2))] 
                demodulated_preamble = self.demodulate_symbols(preamble_symbol_signal)
                print('Demodulated preamble: ', demodulated_preamble)
                if demodulated_preamble == correct_demodulated_preamble:
                    payload_index = int(i + spc * 2**sf * (preamble_number + 4.25))
                    return payload_index
        return -1


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
        self._base_signals = self.generate_base_signals()


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
        self._base_signals = self.generate_base_signals()

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
        self._base_signals = self.generate_base_signals()

