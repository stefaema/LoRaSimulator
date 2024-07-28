import enum
import numpy as np
import matplotlib.pyplot as plt

class LoraModulator():
    def validate_parameters(self):
        '''
        This function validates the parameters of the LoRa modulation.
        '''
        if self._spreading_factor not in range(7, 13):
            raise ValueError('The spreading factor must be an integer between 7 and 12.')
        if self._bandwidth not in [125e3, 250e3, 500e3]:
            print('WARNING: Bandwidth normally is 125 kHz, 250 kHz or 500 kHz.')
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
    
    def generate_timeline(self, symbols_number = 1):
        '''
        This method generates a discrete timeline for the transmission of a given number of symbols.
        Parameters:
        symbols_number: number of symbols to be transmitted
        '''
        # spreading_factor: spreading factor
        # bandwidth: bandwidth
        # samples_per_chip: samples per chip

        # Number of evenly-spaced time samples.
        # A plus one has to be added in order correctly generate the timeline
        num_time_samples= (2**self.spreading_factor) * self.samples_per_chip * symbols_number + 1

        timeline =  np.linspace(0, self._symbol_duration * symbols_number, num_time_samples)

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
        # We only need a single symbol period to generate the frequency evolution
        timeline = timeline[:(len(timeline) // len(symbols))]
        
        frequency_slope = self._frequency_slope

        frequency_evolution = np.array([])
        
        for symbol in symbols:
            
            y_intercept = symbol * ( self.bandwidth / (2**self.spreading_factor))
            instantaneous_frequency = ( y_intercept + frequency_slope * (timeline) ) % self.bandwidth
            frequency_evolution = np.concatenate((frequency_evolution, instantaneous_frequency))

        return frequency_evolution
    
    def generate_signal(self, timeline, symbols):
        '''
        This function generates the complex signal for a given set of symbols, with the given timeline as input.
        Parameters:
        timeline: discrete timeline
        symbols: array of symbolsp
        '''
        # We only need a single symbol period to generate the signal
        timeline = timeline[:(len(timeline) // len(symbols))]

        # Multiplies by 0.5 because of the integration
        integrated_frequency_slope = np.float64(self.bandwidth / self._symbol_duration * 0.5)

        power_normalizer = 1 / np.sqrt(2**self.spreading_factor * self.samples_per_chip)

        signal = np.array([])

        for symbol in symbols:
            y_intercept = symbol * ( self.bandwidth / (2**self.spreading_factor))
            instantaneous_frequency = ( y_intercept + integrated_frequency_slope * (timeline) ) % self.bandwidth
            phase = 2 * np.pi * instantaneous_frequency * timeline
            signal_sample = power_normalizer * np.exp(1j * phase)
            signal = np.concatenate((signal, signal_sample))
    
        return signal
    
    def modulate_symbols(self, symbols, also_compute_frequency_evolution = False, plot = False):
        '''
        This function generates the timeline, the complex signal for a given array of symbols and the frequency evolution if it is also needed.
        Parameters:
        symbols: array of symbols
        also_compute_frequency_evolution: boolean to indicate if the non-integrated frequency evolution should be computed
        '''
        timeline = self.generate_timeline(len(symbols))
        signal = self.generate_signal(timeline, symbols)

        if also_compute_frequency_evolution:
            frequency_evolution = self.generate_frequency_evolution(timeline, symbols)
            return timeline, frequency_evolution, signal
        
        return timeline, signal
    
    def generate_modulation_plots(self, symbols, timeline, frequency_evolution, signal):
        '''
        This function generates the plots for a graphical representation of the modulation process.
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

    def calculate_parameters(self):
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
        self.calculate_parameters()

    @property
    def bandwidth(self):
        return self._bandwidth
    @bandwidth.setter
    def bandwidth(self, bandwidth):
        if bandwidth not in [125e3, 250e3, 500e3]:
            print('WARNING: Bandwidth normally is 125 kHz, 250 kHz or 500 kHz.')
        self._bandwidth = bandwidth
        self.calculate_parameters()

    @property
    def samples_per_chip(self):
        return self._samples_per_chip
    @samples_per_chip.setter
    def samples_per_chip(self, samples_per_chip):
        if samples_per_chip < 1 or not isinstance(samples_per_chip, int):
            raise ValueError('The number of samples per chip must be an integer greater than 0.')
        self._samples_per_chip = samples_per_chip

lora_modulator = LoraModulator(7, 125000)
