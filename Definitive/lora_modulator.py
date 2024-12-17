import numpy as np
from lora_artifacts import LoraReservedArtifacts

class LoraModulator:
    def __init__(self, spreading_factor: int, bandwidth: float, samples_per_chip: int, preamble_length: int, verbosity: str = 'Detailed'):
        # Use property setters for validation and initialization
        self.spreading_factor = spreading_factor
        self.bandwidth = bandwidth
        self.samples_per_chip = samples_per_chip
        self.preamble_length = preamble_length
        self.verbosity = verbosity

        self.visualizer = LoraModulatorVisualizer()

        if self.verbosity == 'Detailed':
            print(f"------- LoRa Modulator initialized! -------\n"
                  f"Spreading Factor = {self.spreading_factor}\n"
                  f"Bandwidth = {self.bandwidth}\n"
                  f"Samples Per Chip = {self.samples_per_chip}\n"
                  f"Preamble Units = {self.preamble_length}\n"
                  f"Verbosity = {self.verbosity}")

    def generate_time_axis(self, symbols):
        """
        Generates the time axis (timeline) for the given symbols in the LoRa modulation.

        :param symbols: A list of symbols, which may include LoRaReservedArtifacts.
        :return: The timeline of the LoRa modulation as a NumPy array.
        """
        symbol_duration = 2 ** self.spreading_factor / self.bandwidth
        chips_number = 2 ** self.spreading_factor
        num_symbols = len(symbols)

        # Adjust for quarter downchirp
        if LoraReservedArtifacts.QUARTER_DOWNCHIRP in symbols:
            num_symbols -= 0.75

        modulation_duration = num_symbols * symbol_duration
        samples_per_symbol = self.samples_per_chip * chips_number
        samples_per_modulation = int(samples_per_symbol * num_symbols)
        return np.linspace(0, modulation_duration, samples_per_modulation, endpoint=False)

    def generate_instantaneous_frequency(self, symbols):
        """
        Generates the instantaneous frequency evolution for given symbols.

        :param symbols: A list of symbols, which may include reserved artifacts.
        :return: The instantaneous frequency as a NumPy array.
        """
        sf = self.spreading_factor
        spc = self.samples_per_chip
        cps = 2 ** sf

        symbol_lengths = []
        symbol_params = []

        for symbol in symbols:
            if not isinstance(symbol, (int, LoraReservedArtifacts)):
                raise ValueError(f"Invalid symbol: {symbol}")
            if isinstance(symbol, LoraReservedArtifacts):
                dur_factor, slope_sign, sym_val = symbol.duration_factor, symbol.slope_sign, symbol.symbol_val
            else:
                dur_factor, slope_sign, sym_val = 1.0, 1.0, symbol

            symbol_lengths.append(int(spc * cps * dur_factor))
            symbol_params.append((sym_val, slope_sign))

        total_samples = sum(symbol_lengths)
        frequency_evolution = np.zeros(total_samples, dtype=float)
        current_pos = 0

        for (sym_val, slope_sign), length in zip(symbol_params, symbol_lengths):
            k = np.arange(length)
            y_intercept = sym_val * (self.bandwidth / cps)
            freq_slice = y_intercept + slope_sign * (k * (self.bandwidth / (cps * spc)))
            if slope_sign > 0:
                freq_slice %= self.bandwidth
            frequency_evolution[current_pos:current_pos + length] = freq_slice
            current_pos += length

        return frequency_evolution

    def generate_instantaneous_phase(self, symbols):
        """
        Generates the instantaneous phase evolution for given symbols.

        :param symbols: A list of symbols (integers or reserved artifacts).
        :return: The instantaneous phase as a NumPy array.
        """
        sf = self.spreading_factor
        spc = self.samples_per_chip
        cps = 2 ** sf

        symbol_lengths = []
        symbol_params = []

        for symbol in symbols:
            if not isinstance(symbol, (int, LoraReservedArtifacts)):
                raise ValueError(f"Invalid symbol: {symbol}")
            if isinstance(symbol, LoraReservedArtifacts):
                dur_factor, slope_sign, sym_val = symbol.duration_factor, symbol.slope_sign, symbol.symbol_val
            else:
                dur_factor, slope_sign, sym_val = 1.0, 1.0, symbol

            symbol_lengths.append(int(cps * spc * dur_factor))
            symbol_params.append((sym_val, slope_sign))

        total_samples = sum(symbol_lengths)
        phase_evolution = np.zeros(total_samples)
        current_pos = 0

        for (sym_val, slope_sign), length in zip(symbol_params, symbol_lengths):
            k = np.arange(length)
            term1 = 2 * np.pi * (sym_val + slope_sign * (k / (2 * spc))) * (k / (cps * spc))
            term2 = -2 * np.pi * ((k / spc) - (cps - sym_val)) * (k / spc > (cps - sym_val)).astype(float)
            phase_evolution[current_pos:current_pos + length] = term1 + term2
            current_pos += length

        return phase_evolution

    def generate_signal(self, symbols):
        """
        Generates the LoRa signal for the given symbols.

        :param symbols: A list of symbols (integers or reserved artifacts).
        :return: The modulated LoRa signal as a NumPy array.
        """
        phase = self.generate_instantaneous_phase(symbols)
        coefficient = 1 / np.sqrt(2 ** self.spreading_factor * self.samples_per_chip)
        return coefficient * np.exp(1j * phase)

    def generate_package(self, payload: list, is_header_explicit: bool = False):
        """
        Generates a full LoRa package with a preamble and optional explicit header.

        :param payload: The payload symbols.
        :param is_header_explicit: Whether to include the explicit header.
        :return: The complete LoRa package signal as a NumPy array.
        """
        preamble = [LoraReservedArtifacts.FULL_UPCHIRP] * self.preamble_length
        sfd = [LoraReservedArtifacts.FULL_UPCHIRP] * 2 + \
              [LoraReservedArtifacts.FULL_DOWNCHIRP] * 2 + \
              [LoraReservedArtifacts.QUARTER_DOWNCHIRP]

        symbols = preamble + sfd + ([len(payload)] if is_header_explicit else []) + payload
        signal = self.generate_signal(symbols)

        if self.verbosity != 'Simple':
            sps = self.samples_per_chip * (2 ** self.spreading_factor)
            indexes = {
                'preamble_end': len(preamble) * sps - 1,
                'sfd_end': len(preamble + sfd) * sps - 1,
                'header_end': (len(preamble + sfd) + 1) * sps - 1 if is_header_explicit else None,
            }
            self.visualizer.plot_package(payload, is_header_explicit, {
                'sf': self.spreading_factor, 'bw': self.bandwidth,
                'spc': self.samples_per_chip, 'pl': self.preamble_length
            }, indexes, self.generate_time_axis(symbols), self.generate_instantaneous_frequency(symbols), signal, self.verbosity)

        return signal

    # Properties
    @property
    def spreading_factor(self):
        return self._spreading_factor

    @spreading_factor.setter
    def spreading_factor(self, value):
        if value not in [7, 8, 9, 10, 11, 12]:
            raise ValueError("Spreading factor must be 7, 8, 9, 10, 11, or 12.")
        self._spreading_factor = value

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        if value not in [125e3, 250e3, 500e3]:
            raise ValueError("Bandwidth must be 125e3, 250e3, or 500e3 Hz.")
        self._bandwidth = value

    @property
    def samples_per_chip(self):
        return self._samples_per_chip

    @samples_per_chip.setter
    def samples_per_chip(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Samples per chip must be a positive integer.")
        self._samples_per_chip = value

    @property
    def preamble_length(self):
        return self._preamble_length

    @preamble_length.setter
    def preamble_length(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Preamble length must be a non-negative integer.")
        self._preamble_length = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if value not in ['Simple', 'Detailed', 'Compact']:
            raise ValueError("Verbosity must be 'Simple', 'Compact', or 'Detailed'.")
        self._verbosity = value

        
class LoraModulatorVisualizer():
    import matplotlib.pyplot as plt
    def __init__(self):
        pass
    def plot_package(self,    
                              payload_symbols: list,
                              is_explicit_header: bool,
                              parameters: dict,
                              indexes: dict,
                              package_time_axis: np.ndarray, 
                              package_instantaneous_frequency: np.ndarray, 
                              package_signal: np.ndarray, 
                              verbosity: str = 'Detailed'):
        """
        TODO: Add docstring
        """
        if verbosity == 'Simple':
            return
        from matplotlib import pyplot as plt
        import numpy as np
        #plt.style.use('seaborn')  # Estilo más profesional
        
        # Definir los índices de las secciones
        preamble_end = indexes['preamble_end']
        sfd_end = indexes['sfd_end']
        header_end = indexes['header_end'] if is_explicit_header else None
        payload_end = len(package_signal) - 1

        # Calcular los índices de inicio de cada sección
        preamble_start = 0
        sfd_start = preamble_end + 1
        header_start = sfd_end + 1 if is_explicit_header else None
        payload_start = header_end + 1 if is_explicit_header else sfd_end + 1

        # Mostrar información en consola
        if verbosity == 'Detailed':
            print("------------ Lora Modulator Visualizer -----------")
            print(f"Transmitting package with:")
            print(f"{len(payload_symbols)} Payload symbol(s): {payload_symbols}")
            print(f"Explicit Header: {is_explicit_header}")
            print(f"Spreading factor: {parameters['sf']}")
            print(f"Bandwidth: {parameters['bw']}")
            print(f"Samples per chip (Oversampling Factor): {parameters['spc']}")
            print(f"Preamble + SFD Structure: {parameters['pl']}+2 Upchirps, 2 Downchirps, 1 Quarter Downchirp")

        # Determinar los rangos que se van a plotear
        if verbosity != 'Detailed':
            # Sólo el payload
            start_idx = payload_start
            end_idx = payload_end
            plot_only_payload = True
        else:
            # Desde el principio hasta el final del payload
            start_idx = 0
            end_idx = payload_end
            plot_only_payload = False

        # Extraer porciones relevantes
        plot_time = package_time_axis[start_idx:end_idx]
        plot_freq = package_instantaneous_frequency[start_idx:end_idx]
        plot_signal = package_signal[start_idx:end_idx]

        # Ajustar índices de las secciones a la ventana actual
        def adjust_idx(idx):
            return idx - start_idx if idx is not None else None

        adj_preamble_end = adjust_idx(preamble_end if not plot_only_payload else None)
        adj_sfd_end = adjust_idx(sfd_end if not plot_only_payload else None)
        adj_header_end = adjust_idx(header_end if is_explicit_header and not plot_only_payload else None)
        # payload end es el final del plot, ya está ajustado indirectamente.

        # Asignar colores y nombres a cada segmento
        segment_info = []
        if not plot_only_payload:
            # Tenemos varios segmentos a mostrar
            segment_info.append(('Preamble', 0, adj_preamble_end, 'tab:green'))
            segment_info.append(('SFD', adj_preamble_end+1 if adj_preamble_end is not None else None, adj_sfd_end, 'tab:orange'))
            if is_explicit_header:
                segment_info.append(('Header', adj_sfd_end+1, adj_header_end, 'tab:purple'))
            # Payload empieza en header_end+1 si header existe, sino en sfd_end+1
            payload_segment_start = (adj_header_end+1 if is_explicit_header else adj_sfd_end+1)
            segment_info.append(('Payload', payload_segment_start, len(plot_time)-1, 'tab:blue'))
        else:
            # Sólo payload
            segment_info.append(('Payload', 0, len(plot_time)-1, 'tab:blue'))

        fig, axs = plt.subplots(3, 1, figsize=(18, 10))
        fig.suptitle("LoRa Modulation Process", fontsize=16)

        # Plot de frecuencia
        axs[0].plot(plot_time, plot_freq, color='grey', linestyle='--', linewidth=1)
        for name, seg_start, seg_end, color in segment_info:
            if seg_start is not None and seg_end is not None and seg_start <= seg_end:
                axs[0].plot(plot_time[seg_start:seg_end+1], plot_freq[seg_start:seg_end+1], color=color, label=name)
        axs[0].set_title("Instantaneous Frequency Evolution", fontsize=14)
        axs[0].set_xlabel("Time (s)", fontsize=12)
        axs[0].set_ylabel("Frequency (Hz)", fontsize=12)
        axs[0].grid(True)
        axs[0].legend()

        # Plot de la parte real
        for name, seg_start, seg_end, color in segment_info:
            if seg_start is not None and seg_end is not None and seg_start <= seg_end:
                axs[1].plot(plot_time[seg_start:seg_end+1], plot_signal.real[seg_start:seg_end+1], color=color, label=name)
        axs[1].set_title("Real Part of the Modulated Signal", fontsize=14)
        axs[1].set_xlabel("Time (s)", fontsize=12)
        axs[1].set_ylabel("Amplitude", fontsize=12)
        axs[1].grid(True)
        axs[1].legend()

        # Plot de la parte imaginaria
        for name, seg_start, seg_end, color in segment_info:
            if seg_start is not None and seg_end is not None and seg_start <= seg_end:
                axs[2].plot(plot_time[seg_start:seg_end+1], plot_signal.imag[seg_start:seg_end+1], color=color, label=name)
        axs[2].set_title("Imaginary Part of the Modulated Signal", fontsize=14)
        axs[2].set_xlabel("Time (s)", fontsize=12)
        axs[2].set_ylabel("Amplitude", fontsize=12)
        axs[2].grid(True)
        axs[2].legend()

        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()
        plt.show()
        



