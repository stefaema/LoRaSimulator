from lora_modem import LoraModulator, LoraDemodulator, LoraSynchronizer
import numpy as np
def ciclify(signal):
    """Shifts the signal in a cyclical nature to simulate a real-world scenario where the signal's first sample isn't alligned to the first received one."""
    rx_buffer = np.array([signal, signal, signal]).flatten()
    roll_factor = np.random.randint(0, len(rx_buffer)//2)
    rx_buffer = np.roll(rx_buffer, roll_factor)

    return rx_buffer
sf, bw, spc, preamb_num = 7, 125e3, 1, 8
mod = LoraModulator(sf, bw, spc)
demod = LoraDemodulator(sf, bw, spc)
payload = [30,60,90]
pkg_freq_evo =mod.debug_modulate_explicit_package( preamb_num, payload)[6]
signal = mod.modulate_explicit_package(preamb_num, payload)
from matplotlib import pyplot as plt
plt.figure(figsize=(100,10))
plt.title("Frequency evolution of the package for SF=7, BW=125kHz, CR=1, Preamb=8, Payload=[30,60,90]")
plt.xlabel("Samples")
plt.ylabel("Frequency (Hz)")
pkg_freq_evo = ciclify(pkg_freq_evo)
plt.plot(pkg_freq_evo)


plt.show()
