from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

class LoraReservedArtifacts(Enum):
    FULL_UPCHIRP = -1
    FULL_DOWNCHIRP = -2
    QUARTER_DOWNCHIRP = -3



class LoraModulator():
    """Class that implements the modulation of LoRa signals."""
    def validate_parameters(self):
        '''
        This function validates the parameters of the LoRa modulation.
        '''
        if self._spreading_factor not in range(7, 13) or not isinstance(self._spreading_factor, int):
            raise ValueError('The spreading factor must be an integer between 7 and 12.')
        if  self._bandwidth <= 0 or not isinstance(self._bandwidth, (int, float)):
            raise ValueError('The bandwidth must be a positive integer or float. Tipically, it is 125e3 Hz, 250e3 Hz or 500e3 Hz.')
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

        whole_frequency_evolution = np.array(whole_frequency_evolution)
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

        whole_phase_evolution = np.array(whole_phase_evolution)
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
        signal = np.array([self._signal_coefficient * np.exp(1j * phase) for phase in instantaneous_phase])
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
        if len(payload) > 2**self._spreading_factor:
            raise ValueError('The payload length must be less than 2^SF.')
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



class LoraDemodulator():
    """Class that implements the demodulation of LoRa signals."""
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


    def demodulate_symbol(self, symbol_signal, base_fn = 'downchirp', return_magnitude = False):
        '''
        This function demodulates a symbol in the LoRa modulation.

        Parameters:

        symbol_signal (np.array): The signal of the symbol in the modulation.
        base_fn (np.array): A string indicating the base function of the symbol in the modulation that shall be used.

        Returns:

        symbol (int): The symbol of the LoRa modulation.
        '''
        base_signal = self._base_signals.get(base_fn)
        if base_signal is None:
            raise ValueError(f'The base function {base_fn} is not recognized. It must be either "downchirp", "upchirp" or "quarter_upchirp".')
        dechirped_signal = [symbol_signal[i] * base_signal[i] for i in range(len(symbol_signal))]
        correlation = np.fft.fft(dechirped_signal)
        correlation = self.adjust_correlation(correlation)
        symbol = np.argmax(correlation) % 2**self._spreading_factor
        
        if return_magnitude:
            return symbol, correlation[symbol]
        return symbol

    def demodulate_symbols(self, signal, base_fn = 'downchirp', return_magnitude = False):
        '''
        This function demodulates the symbols in the LoRa modulation payload, given that the received signal is already synchronized.

        Parameters:

        signal (np.array): The signal of the received LoRa modulation.
        base_fn (np.array): A string indicating the base function to be used in the demodulation.

        Returns:

        symbols (list): The list of symbols in the LoRa modulation.
        '''
        spc = self._samples_per_chip
        sf = self._spreading_factor
        symbols = []
        if not return_magnitude:
            for i in range(0, len(signal), spc * 2**sf):
                symbol_signal = signal[i:i + spc * 2**sf]
                symbol = self.demodulate_symbol(symbol_signal, base_fn)
                symbols.append(symbol)
            return symbols
        else:
            magnitudes = []
            for i in range(0, len(signal), spc * 2**sf):
                symbol_signal = signal[i:i + spc * 2**sf]
                symbol, magnitude = self.demodulate_symbol(symbol_signal, base_fn, return_magnitude)
                symbols.append(symbol)
                magnitudes.append(magnitude)
            return symbols, magnitudes



class LoraSynchronizer():
    """Class that implements the synchronization of LoRa signals."""
    def validate_parameters(self, spreading_factor, samples_per_chip, demodulator, preamble_number):
        if spreading_factor not in range(7, 13) or not isinstance(spreading_factor, int):
            raise ValueError('The spreading factor must be an integer between 7 and 12.')
        if samples_per_chip < 1 or not isinstance(preamble_number, int):
            raise ValueError('Samples per chip must be a positive integer')
        if not isinstance(demodulator, LoraDemodulator):
            raise ValueError('Demodulator must be an instance of LoraDemodulator')
        if preamble_number < 1 or not isinstance(preamble_number, int):
            raise ValueError('Preamble number must be a positive integer')
        
    def __init__(self, spreading_factor, samples_per_chip, demodulator, preamble_number):
        self.validate_parameters(spreading_factor, samples_per_chip, demodulator, preamble_number)
        self._spreading_factor = spreading_factor
        self._samples_per_chip = samples_per_chip
        self._demodulator = demodulator 
        self._preamble_number = preamble_number

    def _detect_chirp(self, signal_segment, chirp_type):
        """Helper function to detect specific chirp types in a given signal segment."""
        chirp_type = 'upchirp' if chirp_type == 'downchirp' else 'downchirp'
        return self._demodulator.demodulate_symbol(signal_segment, chirp_type) == 0

    def _get_samples_per_symbol(self):
        return self._samples_per_chip * 2 ** self._spreading_factor

    def synchronize_rx_buffer(self, rx_signal):
        """
        Function that synchronizes a receiver buffer and returns the signal segment allegedly containing the payload of the received package.
        Parameters:
        rx_signal (np.array): The received signal buffer.
        Returns:
        rx_payload_segment (np.array): The synchronized payload segment. Contains the payload of the LoRa package (if the synchronization process was successfull). 
        """
        sps = self._get_samples_per_symbol()
        preamble_found, payload_index, package_length = self._phase1sync(rx_signal)
        #Ignoring of the payload length symbol
        payload_index += sps

        if preamble_found:
            message_samples = (package_length) * sps 
            payload_start = payload_index
            payload_end = payload_index + message_samples
            rx_payload_segment = rx_signal[payload_start: payload_end]
            return rx_payload_segment
        else:
            print('Synchronization failed!')
            return None
    
    def _phase1sync(self, rx_signal):
        sps = self._get_samples_per_symbol()
        print('Synchronization started...')
        print("Phase 1: Searching for upchirps...")
        for i in range(len(rx_signal)):
            # Extract the iterative segment to analyze
            segment = rx_signal[i:i + sps]
            if len(segment) < sps:
                break
            # Check if the segment is an upchirp
            if self._detect_chirp(segment, 'upchirp'):
                # print('Upchirp found at index: ', i)
                # Check if the next symbols are members of the preamble (phase 2)
                preamble_found, payload_index, reconstructed_preamble = self._phase2sync(rx_signal, i)
                
                if preamble_found:
                    
                    offset = 0
                    if self._samples_per_chip > 1:
                        
                        payload_index, offset = self._phase3sync(rx_signal, i, payload_index, reconstructed_preamble)
                        
                    package_length = self._demodulator.demodulate_symbol(rx_signal[payload_index:payload_index + sps], 'downchirp')
                    
                    print('Synchronization successful!\n-----------------------------------------------------------')
                    
                    print('Preamble found at index: ', i + offset)
                    if self._samples_per_chip > 1:
                        print('Refined by an offset of: ', offset)
                    print('Package length: ', package_length)
                    print('Payload starts at index: ', payload_index)
                    upchirps = reconstructed_preamble.count(LoraReservedArtifacts.FULL_UPCHIRP)
                    downchirps = reconstructed_preamble.count(LoraReservedArtifacts.FULL_DOWNCHIRP)
                    print(f'Reconstructed preamble: [{upchirps + 1} upchirps, {downchirps} downchirps]')

                    return True, payload_index, package_length
        return False, -1, -1
    
    def _phase2sync(self, rx_signal, candidate_index):
        sps = self._get_samples_per_symbol()
        
        current_index = candidate_index + sps
        
        reconstructed_preamble = []
        while True:
            segment = rx_signal[current_index:current_index + sps]
            # Check if there is enough samples to analyze. If not, break the loop and therefore the synchronization process (as the preamble is not complete)
            if len(segment) < sps:
                break
            # Check if the segment is an upchirp. If so, append it to the reconstructed preamble and move to the next segment, hoping it is a downchirp.
            if self._detect_chirp(segment, 'upchirp'):
                reconstructed_preamble.append(LoraReservedArtifacts.FULL_UPCHIRP)
                current_index += sps
                continue
            # If the segment is not an upchirp, check if it is a downchirp. If so, append it to the reconstructed preamble and move to the next segment, hoping it is another downchirp.
            elif self._detect_chirp(segment, 'downchirp'):
                reconstructed_preamble.append(LoraReservedArtifacts.FULL_DOWNCHIRP)
                current_index += sps
                alleged_second_downchip = rx_signal[current_index:current_index + sps]
                # If the second segment is not a downchirp, break the loop and therefore the synchronization process (as the preamble is not complete)
                # If the second segment is a downchirp, the preamble is complete (as the preamble ends with two consecutive downchirps) and the synchronization process is successful.
                if self._detect_chirp(alleged_second_downchip, 'downchirp'):
                    reconstructed_preamble.append(LoraReservedArtifacts.FULL_DOWNCHIRP)
                    payload_index = int(current_index + sps * 1.25)
                    print("Phase 2: Ensuring that the candidate upchirp is a preamble member...")
                    print("Preamble allegedly found at index: ", candidate_index)
                    return True, payload_index, reconstructed_preamble
                else:
                    break
            else:
                break
        return False, -1, -1
    
    def _evaluate_offset(self, rx_signal, candidate_index, offset, upchirps, downchirps):
        """
        Evaluate the quality of a given offset for a index synchronization candidate.

        Parameters:
        rx_signal (np.array): The received signal buffer.
        candidate_index (int): The index of the candidate synchronization point.
        offset (int): The offset to evaluate.
        upchirps (int): The number of upchirps in the preamble.
        downchirps (int): The number of downchirps in the preamble.

        Returns:
        mean_magnitude (float): The mean magnitude of the demodulated preamble with the given offset (It is the way we found to rank the different offsets).
        """
        sps = self._get_samples_per_symbol()
        upchirps_index = candidate_index + sps + offset
        downchirps_index = upchirps_index + sps * upchirps
        upchirps_length = sps * upchirps
        downchirps_length = sps * downchirps
        demod = []
        upchirps_segment = rx_signal[upchirps_index:upchirps_index + upchirps_length]
        downchirps_segment = rx_signal[downchirps_index:downchirps_index + downchirps_length]
        upchirp_symbols, upchirp_magnitudes = self._demodulator.demodulate_symbols(upchirps_segment, 'downchirp', return_magnitude=True)
        downchirp_symbols, downchirp_magnitudes = self._demodulator.demodulate_symbols(downchirps_segment, 'upchirp', return_magnitude=True)
        magnitudes = np.concatenate((np.abs(upchirp_magnitudes), np.abs(downchirp_magnitudes)))
        mean_magnitude = np.mean(magnitudes)
        demod.extend(upchirp_symbols)
        demod.extend(downchirp_symbols)
        if demod.count(0) != len(demod):
            return 0
        return mean_magnitude

    def _phase3sync(self, rx_signal, candidate_index, payload_index, reconstructed_preamble):
        """
        Refine the synchronization index by evaluating different offsets and choosing the one that ensures the highest quality synchronization.
        
        Parameters:
        rx_signal (np.array): The received signal buffer.
        candidate_index (int): The index of the candidate synchronization point
        payload_index (int): The index of the payload start.
        reconstructed_preamble (list): The reconstructed preamble.

        Returns:
        payload_index (int): The refined payload start index.
        chosen_offset (int): The chosen offset that ensures the highest quality synchronization.

        """


        sps = self._get_samples_per_symbol()
        print("Phase 3: Refining synchronization index... (SPC > 1 requires it)")
        upchirps = reconstructed_preamble.count(LoraReservedArtifacts.FULL_UPCHIRP)
        downchirps = reconstructed_preamble.count(LoraReservedArtifacts.FULL_DOWNCHIRP)
        offset = 0
        offset_magnitudes = []
        for offset in range(0, self._samples_per_chip):
            mean_magnitude = self._evaluate_offset(rx_signal, candidate_index, offset, upchirps, downchirps)
            offset_magnitudes.append(mean_magnitude)
        offset_magnitudes = np.array(offset_magnitudes)
        # Normalizing the magnitudes
        offset_magnitudes = offset_magnitudes / np.max(offset_magnitudes)
        chosen_offset = np.argmax(offset_magnitudes)
        print("Offset quality measurements: ", offset_magnitudes)
        print("Offset that ensures Highest Quality Synchronization: ", chosen_offset)
        payload_index = chosen_offset + payload_index
        return payload_index, chosen_offset
