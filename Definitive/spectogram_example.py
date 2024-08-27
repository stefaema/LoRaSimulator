import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Generate a sample signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 2, 2 * fs, endpoint=False)  # Time array (2 seconds)
f = 50  # Frequency of the sine wave
signal = np.sin(2 * np.pi * f * t)  # Generate a sine wave

# Generate the spectrogram
frequencies, times, Sxx = spectrogram(signal, fs)

# Plot the spectrogram
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.colorbar(label='Intensity [dB]')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.show()