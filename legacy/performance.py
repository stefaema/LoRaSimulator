from lora_modem import LoraModulator, LoraDemodulator, AnotherSimpleLoraMoDem
import numpy as np
from matplotlib import pyplot as plt
def signal_power(signal):
    # Calculate the signal power
    P = np.mean(np.abs(signal)**2)
    return P

def generate_awgn(SNR, w):
    # Check if SNR is given in dB, DB, or db and convert if necessary
    if isinstance(SNR, str) and SNR.lower().endswith('db'):
        SNR_linear = 10 ** (float(SNR[:-2]) / 10)
    elif isinstance(SNR, int) or isinstance(SNR, float):
        SNR_linear = float(SNR)
    else:
        raise ValueError("SNR must be a number or a string ending with 'dB'.")
    
    if SNR_linear <= 0:
        raise ValueError("SNR must be a positive value.")
    signal_len = len(w)

    # Determine the noise power (variance)
    variance = signal_power(w) / SNR_linear
    std_dev = np.sqrt(variance)
    # Generate noise
    noise = noise = np.sqrt(variance / 2) * (np.random.randn(signal_len) + 1j * np.random.randn(signal_len))
    
    # Add noise to the signal
    noisy_signal = w + noise
    
    return noisy_signal, noise, variance

def generate_all_symbols(sf, spc):
    mod = AnotherSimpleLoraMoDem(sf, 125e3, spc)
    symbol_signals = []
    for i in range(2**sf):
        symbol_signals.append(mod.modulate_symbols([i]))
    return symbol_signals

def generate_BER_SNR_ratio_binary(sf, simulations_number=1000000, spc = 1, snr_range = None):

    # Set up SNR range
    if snr_range is None:
        snr_range = np.arange(-30 + 12 - sf, 2, 1)
    # Initialize MoDem
    bits_per_symbol = sf
    demod = AnotherSimpleLoraMoDem(sf, 125e3, spc)
    symbol_signals = generate_all_symbols(sf, spc)
    # Initialize storage arrays
    SNR_values = []
    BER_values = []
    
        
    # Loop over all SNR values
    
    for snr in snr_range:
        print(f'Processing BER for SNR = {snr} dB')
        bit_errors = 0

        for _ in range(simulations_number):
            # Generate a random message
            message = np.random.randint(0, 2**sf)
            # Modulate the message
            signal = symbol_signals[message]
            # Generate AWGN
            noisy_signal = generate_awgn(f'{snr}dB', signal)[0]
            # Demodulate the signal
            demodulated_message = demod.demodulate_symbols(noisy_signal)[0]

            # XOR the messages to find the bit errors
            bit_errors += bin(demodulated_message ^ message).count('1')
                
        # Calculate BER for this SNR
        BER = bit_errors / (simulations_number*bits_per_symbol)
        SNR_values.append(snr)
        BER_values.append(BER)
        print(f'BER for SNR = {snr} dB is {BER}')
    
    # Convert lists to numpy arrays
    SNR_values = np.array(SNR_values)
    BER_values = np.array(BER_values)

    # Save to a binary .npy file
    np.save(f'Definitive/Measures/BER_SNR_ratio_sf{sf}_spc{spc}.npy', np.vstack((SNR_values, BER_values)))

    print("BER-SNR Metrics successfully generated and saved in binary format for a Spreading Factor of {sf} and samples per chip of {spc}.")

def plot_BER_SNR_from_binary(filename, sf, spc):
    # Load the binary file
    data = np.load(filename)
    SNR_values = data[0, :]
    BER_values = data[1, :]
    # Plot the data
    plt.plot(SNR_values, BER_values, marker='o', linestyle='-', label=f'SF {sf} SPC={spc}')
    plt.xlim([SNR_values.min(), SNR_values.max()])
    plt.yscale('log')  # Logarithmic scale for BER
    plt.xlabel('SNR (dB)')
    plt.xticks(np.arange(-30 + 12 - sf, 2, 1))
    plt.ylabel('BER (log scale)')
    plt.title(f'LoRa MoDem: BER vs. SNR')
    plt.grid(True)

if __name__ == '__main__':
    # Generate BER-SNR metrics for a given SF
    sf = 7
    spc = 1
    snr_range = np.arange(-10,-6)
    
    # Plot the BER-SNR metrics
    #generate_BER_SNR_ratio_binary(sf, spc=spc, snr_range=snr_range)
    plot_BER_SNR_from_binary(f'Definitive/Measures/BER_SNR_ratio_sf{sf}_spc{spc}.npy', sf, spc)
    plt.legend()
    plt.show()