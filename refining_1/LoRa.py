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
        if  not isinstance(self.bandwidth, (int, float)):
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
    
    def generate_timeline(self, symbols):
        '''
        This method generates a discrete timeline for the transmission of a given number of symbols.
        This method is not intended to handle the quarter chirp reserved artifact.
        Parameters:
        symbols_number: number of symbols to be transmitted
        '''
        symbols_number = len(symbols)
        if symbols_number == 0:
            raise ValueError('The symbols array must not be empty. They will the ones to be transmitted.')

        # Number of evenly-spaced time samples.
        # A plus one has to be added in order correctly generate the timeline (afterwards we will remove the last element)
        num_time_samples= int((2**self.spreading_factor) * self.samples_per_chip * symbols_number + 1)

        complete_duration = self._symbol_duration * symbols_number
        timeline =  np.linspace(0, complete_duration, num_time_samples)

        # Remove the last element of the timeline due to the discretization
        timeline = timeline[:-1]

        return timeline
    
    def generate_frequency_evolution(self, timeline, symbols):
        '''
        This function generates the linear frequency evolution for a given set of symbols, with the given timeline as input.
        Parameters:
        timeline: discrete timeline
        symbols: array of symbols to be transmitted
        '''
        samples_per_symbol = int(2**self.spreading_factor * self.samples_per_chip)
        # We only need a single symbol period to generate the frequency evolution
        symbol_timeline = timeline[:samples_per_symbol]
        
        frequency_slope = self._frequency_slope

        frequency_evolution = np.array([])
        
        for symbol in symbols:
            if not isinstance(symbol, LoraReservedArtifacts) and symbol in range(2**self.spreading_factor):
                y_intercept = symbol * ( self.bandwidth / (2**self.spreading_factor))
                instantaneous_frequency = ( y_intercept + frequency_slope * (symbol_timeline) ) % self.bandwidth
            elif symbol == LoraReservedArtifacts.FULL_UPCHIRP:
                y_intercept = 0
                instantaneous_frequency = ( y_intercept + frequency_slope * (symbol_timeline) )
            elif symbol == LoraReservedArtifacts.FULL_DOWNCHIRP:
                y_intercept = self.bandwidth
                instantaneous_frequency = ( self.bandwidth - frequency_slope * (symbol_timeline) )
            elif symbol == LoraReservedArtifacts.QUARTER_DOWNCHIRP:
                y_intercept = self.bandwidth
                # This could be a problem because of the phase disalignment (each symbol goes from 0 to 2*pi)
                quarter_symbol_timeline = symbol_timeline[:samples_per_symbol//4]
                instantaneous_frequency = ( self.bandwidth - frequency_slope * (quarter_symbol_timeline) )
            else:
                raise ValueError('Unknown symbol. This should not happen.')
            frequency_evolution = np.concatenate((frequency_evolution, instantaneous_frequency))

        return frequency_evolution
    
    def generate_signal(self, timeline, symbols):
        '''
        This function gerates the complex signal for a given set of symbols, with the given timeline as input.
        Parameters:
        timeline: discrete timeline
        symbols: array of symbols
        '''
        samples_per_symbol = int(2**self.spreading_factor * self.samples_per_chip)
        # We only need a single symbol period to generate the signal
        symbol_timeline = timeline[:samples_per_symbol]

        # Multiplies by 0.5 because of the integration over time
        integrated_frequency_slope = np.float64(self.bandwidth / self._symbol_duration * 0.5)

        power_normalizer = 1 / np.sqrt(2**self.spreading_factor * self.samples_per_chip)

        signal = np.array([])

        for symbol in symbols:
            if not isinstance(symbol, LoraReservedArtifacts):
                y_intercept = symbol * ( self.bandwidth / (2**self.spreading_factor))
                instantaneous_frequency = ( y_intercept + integrated_frequency_slope * (symbol_timeline) ) % self.bandwidth
                phase = 2 * np.pi * instantaneous_frequency * symbol_timeline
                signal_sample = power_normalizer * np.exp(1j * phase)
            elif symbol == LoraReservedArtifacts.FULL_UPCHIRP:
                y_intercept = 0
                instantaneous_frequency = ( y_intercept + integrated_frequency_slope * (symbol_timeline) ) % self.bandwidth
                phase = 2 * np.pi * instantaneous_frequency * symbol_timeline
                signal_sample = power_normalizer * np.exp(1j * phase)
            elif symbol == LoraReservedArtifacts.FULL_DOWNCHIRP:
                y_intercept = 0
                instantaneous_frequency = ( self.bandwidth - integrated_frequency_slope * (symbol_timeline) )
                phase = 2 * np.pi * instantaneous_frequency * symbol_timeline
                signal_sample = power_normalizer * np.exp(1j * phase)
            elif symbol == LoraReservedArtifacts.QUARTER_DOWNCHIRP:
                y_intercept = 0
                # This could be a problem because of the phase disalignment (each symbol goes from 0 to 2*pi)
                quarter_symbol_timeline = symbol_timeline[:int(len(symbol_timeline))//4]
                instantaneous_frequency = ( self.bandwidth - integrated_frequency_slope * (quarter_symbol_timeline) )
                phase = 2 * np.pi * instantaneous_frequency * quarter_symbol_timeline
                signal_sample = power_normalizer * np.exp(1j * phase)
            else:
                raise ValueError('Unknown symbol. Thsi should not happen.')

            signal = np.concatenate((signal, signal_sample))
    
        return signal
    
    def modulate_symbols(self, symbols, also_compute_frequency_evolution = False):
        '''
        This function generates the timeline, the complex signal for a given array of symbols and the frequency evolution if it is also needed.
        
        Parameters:
        symbols: array of symbols
        also_compute_frequency_evolution: boolean to indicate if the non-integrated frequency evolution should be computed
        
        Returns:
        timeline: discrete timeline
        frequency_evolution: frequency evolution of the signal (if also_compute_frequency_evolution is True)
        signal: complex signal to be transmitted

        '''
        if symbols == []:
            raise ValueError('The symbols array must not be empty. They will the ones to be transmitted.')
        
        for symbol in symbols:
            if symbol not in range(2**self.spreading_factor) and not isinstance(symbol, LoraReservedArtifacts):
                raise ValueError(f'The spreading factor set for the LoRa modulator is {self.spreading_factor}. The symbols must be integers between 0 and {2**self.spreading_factor - 1} or reserved LoRa artifacts.')
        

        timeline = self.generate_timeline(symbols)
        signal = self.generate_signal(timeline, symbols)

        if also_compute_frequency_evolution:
            frequency_evolution = self.generate_frequency_evolution(timeline, symbols)
            timeline = timeline[:len(frequency_evolution)]
            return timeline, frequency_evolution, signal
        
        timeline = timeline[:len(signal)]
        return timeline, signal
    
    def generate_implicit_package(self, preamble_number, payload, also_compute_frequency_evolution = False):
        '''
        This function generates the implicit package for a given preamble number and payload.
        
        Parameters:
        preamble_number: number of preamble symbols
        payload: array of symbols to be transmitted
        
        Returns:
        package: array with the preamble symbols and the payload
        '''
        package = []
        for i in range(preamble_number):
            package.append(LoraReservedArtifacts.FULL_UPCHIRP)
        for i in range(2):
            package.append(LoraReservedArtifacts.FULL_UPCHIRP)
        for i in range(2):
            package.append(LoraReservedArtifacts.FULL_DOWNCHIRP)
        package.append(LoraReservedArtifacts.QUARTER_DOWNCHIRP)

        package.extend(payload)
        print(package)
        
        return self.modulate_symbols(package, also_compute_frequency_evolution)



    def generate_modulation_plots(self, symbols, timeline, frequency_evolution, signal):
        '''
        This function generates the plots for a graphical representation of the modulation process.
        
        Parameters:
        symbols: array of symbols
        timeline: discrete timeline
        frequency_evolution: frequency evolution of the signal
        signal: complex signal to be transmitted|

        Returns:
        figure: matplotlib figure with the plots

        '''
        figure, axes = plt.subplots(2, 1, figsize=(20, 12))
        too_many_symbols = len(symbols) > 5
        symbols_txt = str(symbols) if not too_many_symbols else str(symbols[:4]) + '...'
        figure.suptitle('LoRa Modulation of the symbols: '+symbols_txt, fontsize=20)
        axes[0].plot(timeline, frequency_evolution)
        axes[0].set_title('Frequency evolution across time of the signal to be transmitted')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[1].plot(timeline, np.real(signal))
        axes[1].plot(timeline, np.imag(signal))
        axes[1].set_title('Real and imaginary parts of the signal to be transmitted')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        return figure

    def _calculate_parameters(self):
        '''
        This function re-calculates the parameters of the LoRa modulation.
        '''
        self._chips_number = 2**self.spreading_factor
        self._symbol_duration = 2**self.spreading_factor / self.bandwidth
        self._frequency_slope = self.bandwidth / self._symbol_duration

    

    @property
    def spreading_factor(self):
        return self._spreading_factor
    @spreading_factor.setter
    def spreading_factor(self, spreading_factor):
        if spreading_factor not in range(7, 13):
            raise ValueError('The spreading factor must be an integer between 7 and 12.')
        self._spreading_factor = spreading_factor
        self._calculate_parameters()

    @property
    def bandwidth(self):
        return self._bandwidth
    @bandwidth.setter
    def bandwidth(self, bandwidth):
        if  not isinstance(self.bandwidth, (int, float)):
            raise ValueError('The bandwidth must be an integer or float. Normally, it is 125e3 Hz, 250e3 Hz or 500e3 Hz.')
        if bandwidth not in [125e3, 250e3, 500e3, 125000, 25000, 500000]:
            print('WARNING: Bandwidth typically takes values of 125e3 Hz, 250e3 Hz, or 500e3 Hz.')
        self._bandwidth = bandwidth
        self._calculate_parameters()

    @property
    def samples_per_chip(self):
        return self._samples_per_chip
    @samples_per_chip.setter
    def samples_per_chip(self, samples_per_chip):
        if samples_per_chip < 1 or not isinstance(samples_per_chip, int):
            raise ValueError('The number of samples per chip must be an integer greater than 0.')
        self._samples_per_chip = samples_per_chip


lora_modulator = LoraModulator(8, 125e3, 10)
symbols = [65, 10, 35, 9]
package = lora_modulator.generate_implicit_package(8, symbols, True)

lora_modulator.generate_modulation_plots(symbols, *package)
plt.show()