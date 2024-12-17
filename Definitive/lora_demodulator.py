from lora_artifacts import LoraReservedArtifacts
from lora_modulator import LoraModulator
import numpy as np

import numpy as np
from lora_artifacts import LoraReservedArtifacts


class LoraDemodulator:
    def __init__(self, spreading_factor: int, bandwidth: float, samples_per_chip: int, verbosity: str = "Detailed"):
        """
        Initialize the LoRa demodulator.

        :param spreading_factor: Spreading factor of the LoRa signal.
        :param bandwidth: Bandwidth of the LoRa signal.
        :param samples_per_chip: Number of samples per chip.
        :param verbosity: Verbosity level for the demodulator. Options: "Simple", "Detailed", "Compact", and "Magnitudes".
        """
        # Directly initialize backing fields to avoid premature access
        self._samples_per_chip = samples_per_chip
        self._bandwidth = bandwidth

        self.spreading_factor = spreading_factor
        self.bandwidth = bandwidth
        self.samples_per_chip = samples_per_chip
        self.verbosity = verbosity

        # Precompute values
        self.chips_number = 2 ** self.spreading_factor
        self.samples_per_symbol = self.samples_per_chip * self.chips_number
        self.signal_coefficient = 1 / np.sqrt(self.samples_per_symbol)

        # Generate base signals and visualizer
        self.base_signals = self._generate_base_signals()
        self.visualizer = LoraDemodulatorVisualizer()

        # Optional verbose output
        if self.verbosity == "Detailed":
            print(f"-------- LoRa Demodulator initialized! --------\n"
                  f"Bandwidth = {self.bandwidth} Hz\n"
                  f"Spreading Factor = {self.spreading_factor}\n"
                  f"Samples Per Chip = {self.samples_per_chip}\n"
                  f"Verbosity = {self.verbosity}")

    def _generate_chirp(self, slope_sign: int, duration_factor: float = 1.0):
        """
        Generate a chirp signal (upchirp or downchirp).

        :param slope_sign: +1 for upchirp, -1 for downchirp.
        :param duration_factor: Duration factor (e.g., 0.25 for quarter chirps).
        :return: Chirp signal as a NumPy array.
        """
        num_samples = int(self.samples_per_symbol * duration_factor)
        k = np.arange(num_samples)
        phase = slope_sign * 2 * np.pi * (k / (2 * self.samples_per_chip)) * (k / (self.chips_number * self.samples_per_chip))
        return self.signal_coefficient * np.exp(1j * phase)

    def _generate_base_signals(self):
        """
        Generate the base signals for the demodulation process.

        :return: Dictionary of base signals.
        """
        return {
            "upchirp": self._generate_chirp(+1),
            "downchirp": self._generate_chirp(-1),
            "quarter_upchirp": self._generate_chirp(+1, 0.25),
        }

    def adjust_fft(self, fft: np.ndarray):
        """
        Adjust the FFT output for oversampling.

        :param fft: FFT array from the dechirped signal.
        :return: Adjusted FFT array.
        """
        if self.samples_per_chip == 1:
            return fft

        fft_ideal_length = self.chips_number
        spc = self.samples_per_chip
        adjusted_fft = fft[:fft_ideal_length]
        oversampled_fft = fft[(spc - 1) * fft_ideal_length: spc * fft_ideal_length]
        return adjusted_fft + oversampled_fft

    def demodulate_symbol(self, symbol_signal: np.ndarray, base_fn: str = "downchirp"):
        """
        Demodulate a single symbol.

        :param symbol_signal: Received signal for the symbol.
        :param base_fn: Base function to use ("downchirp", "upchirp", or "quarter_upchirp").
        :return: Tuple (dechirped signal, FFT, decoded symbol, magnitude).
        """
        base_signal = self.base_signals.get(base_fn)
        if base_signal is None:
            raise ValueError(f"Base function '{base_fn}' is not recognized.")
        if len(symbol_signal) != len(base_signal):
            raise ValueError("Symbol signal and base function sizes don't match.")

        dechirped_signal = symbol_signal * base_signal
        fft = np.fft.fft(dechirped_signal)
        fft = self.adjust_fft(fft)

        symbol = np.argmax(np.abs(fft)) % self.chips_number
        magnitude = np.abs(fft[symbol])
        return dechirped_signal, fft, symbol, magnitude

    def demodulate_symbols(self, signal: np.ndarray, base_fn: str = "downchirp"):
        """
        Demodulate a sequence of symbols.

        :param signal: Received signal for the symbols.
        :param base_fn: Base function to use for demodulation.
        :return: List of decoded symbols (optionally with magnitudes, dechirped signals, and FFTs).
        """
        num_symbols = len(signal) // self.samples_per_symbol
        symbols, magnitudes, dechirped_signals, ffts = [], [], [], []

        for i in range(num_symbols):
            start = i * self.samples_per_symbol
            end = start + self.samples_per_symbol
            symbol_signal = signal[start:end]

            dchrp_signal, fft, symbol, magnitude = self.demodulate_symbol(symbol_signal, base_fn)
            symbols.append(symbol)
            magnitudes.append(magnitude)
            dechirped_signals.append(dchrp_signal)
            ffts.append(fft)

        # Visualize results if verbosity is not simple
        if self.verbosity != "Simple" and self.verbosity != "Magnitudes":
            self.visualizer.plot_demodulation(
                symbols, dechirped_signals, ffts, magnitudes, verbosity=self.verbosity
            )

        # Return based on verbosity
        if self.verbosity == "Detailed":
            return symbols, magnitudes, dechirped_signals, ffts
        elif self.verbosity == "Magnitudes":
            return symbols, magnitudes
        return symbols

    # Properties
    @property
    def spreading_factor(self):
        return self._spreading_factor

    @spreading_factor.setter
    def spreading_factor(self, value):
        if value not in [7, 8, 9, 10, 11, 12]:
            raise ValueError("Spreading factor must be 7, 8, 9, 10, 11, or 12.")
        self._spreading_factor = value
        self.chips_number = 2 ** value
        self.samples_per_symbol = self.samples_per_chip * self.chips_number
        self.signal_coefficient = 1 / np.sqrt(self.samples_per_symbol)

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
        if value <= 0:
            raise ValueError("Samples per chip must be greater than 0.")
        self._samples_per_chip = value
        self.samples_per_symbol = self.samples_per_chip * self.chips_number
        self.signal_coefficient = 1 / np.sqrt(self.samples_per_symbol)

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if value not in ["Simple", "Detailed", "Compact", "Magnitudes"]:
            raise ValueError("Verbosity must be 'Simple', 'Detailed', 'Compact', or 'Magnitudes'.")
        self._verbosity = value

    
class LoraDemodulatorVisualizer():
    def __init__(self):
        pass
    def plot_demodulation(self, received_symbols, dechirped_signals, ffts, magnitudes, verbosity):
        """
        Visualizes the dechirped signals and their FFTs in a dynamic, scalable layout.

        :param received_symbols: List of decoded symbols.
        :param dechirped_signals: List of complex dechirped signals.
        :param ffts: List of FFT results for each dechirped signal.
        :param magnitudes: List of the maximum magnitud for each FFT result.
        :param verbosity: Verbosity level for the visualizer. Options: "Simple", "Detailed", "Compact" and "Magnitudes".
        Compact: Prints the received symbols and plots the FFTs only.
        Detailed: Prints the received symbols, plots the dechirped signals and FFTs.
        Simple and Magnitudes should never be used here.
        """
        import matplotlib.pyplot as plt
        print("-------- Lora Demodulator Visualizer --------")
        if verbosity == "Compact":
            print("Received Symbols:", received_symbols)
            fig, ax_fft = plt.subplots(figsize=(18, 5))
            fig.suptitle("LoRa Demodulation Process", fontsize=16)

            for i, fft in enumerate(ffts):
                ax_fft.plot(np.abs(fft), label=f"Symbol {i}: {received_symbols[i]}")

            ax_fft.set_title("FFT of the Dechirped Signals", fontsize=12)
            ax_fft.set_xlabel("Frequency (Hz)", fontsize=10)
            ax_fft.set_ylabel("Magnitude", fontsize=10)
            ax_fft.grid(True)
            ax_fft.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
            plt.show()
            return
        elif verbosity == "Detailed":

            info_message = f"Received and Demodulated {len(received_symbols)} symbol(s): "
            for i in range(len(received_symbols)):
                mag = round(magnitudes[i]*100, 3)
                sym = received_symbols[i]
                info_message += f"\n({mag}% acc) {sym}"
            
            print(info_message)

            n_dechirped = len(dechirped_signals)

            # Use gridspec for dynamic proportional heights
            from matplotlib.gridspec import GridSpec
            from matplotlib import pyplot as plt

            # Set overall figure size
            fig = plt.figure(figsize=(18, 5 + n_dechirped * 2))
            fig.suptitle("LoRa Demodulation Process", fontsize=16)

            # Create a GridSpec layout
            gs = GridSpec(n_dechirped + 1, 1, height_ratios=[1] * n_dechirped + [2], figure=fig)

            # Plot dechirped signals
            for i, dechirped_signal in enumerate(dechirped_signals):
                ax = fig.add_subplot(gs[i, 0])
                dechirped_signal = np.array(dechirped_signal)

                ax.plot(dechirped_signal.real, label="Real Part")
                ax.plot(dechirped_signal.imag, label="Imaginary Part")

                ax.set_title(f"Dechirped Signal for {i}th Received Symbol", fontsize=12)
                ax.set_xlabel("Samples", fontsize=10)
                ax.set_ylabel("Amplitude", fontsize=10)
                ax.grid(True)
                ax.legend()

            # Plot FFTs in the last subplot
            ax_fft = fig.add_subplot(gs[n_dechirped, 0])
            for i, fft in enumerate(ffts):
                ax_fft.plot(np.abs(fft), label=f"Symbol {i}: {received_symbols[i]}")

            ax_fft.set_title("FFT of the Dechirped Signals", fontsize=12)
            ax_fft.set_xlabel("Frequency (Hz)", fontsize=10)
            ax_fft.set_ylabel("Magnitude", fontsize=10)
            ax_fft.grid(True)
            ax_fft.legend()

            # Automatically adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
            plt.show()