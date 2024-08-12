from lora_modem import LoraModulator, LoraDemodulator
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats

def signal_power(signal):
    # Calculate the signal power
    P = np.mean(np.abs(signal)**2)
    return P


def generate_awgn(SNR, w):
    # Check if SNR is given in dB, DB, or db and convert if necessary
    if isinstance(SNR, str) and SNR.lower().endswith('db'):
        # Extract the numeric part and convert from dB to linear scale
        SNR_value = float(SNR[:-2])  # Remove the last two characters and convert to float
        SNR_linear = 10 ** (SNR_value / 10)
    else:
        # Assume SNR is already in linear scale
        SNR_linear = SNR
    
    # Determine the noise power (variance)
    variance = signal_power(w) / SNR_linear
    std_dev = variance ** 0.5
    # Generate noise
    noise_real = np.random.normal(0, std_dev, len(w))
    noise_imag = np.random.normal(0, std_dev, len(w))
    noise = noise_real + 1j * noise_imag
    
    # Add noise to the signal
    noisy_signal = w + noise
    
    return noisy_signal, noise, variance

def generate_all_symbols(sf):
    mod = LoraModulator(sf, 125e3, 1)
    symbol_signals = []
    for i in range(2**sf):
        symbol_signals.append(mod.modulate_symbols([i])[2])
    return symbol_signals


def generate_SER_SNR_ratio_binary(sf):
    sims = 100000
    # Set up SNR range
    snr_range = np.arange(-30 + 12 - sf, 2, 1)
    # Initialize MoDem
    
    demod = LoraDemodulator(sf, 125e3, 1)
    symbol_signals = generate_all_symbols(sf)
    # Initialize storage arrays
    SNR_values = []
    SER_values = []

    # Loop over all SNR values
    for snr in snr_range:
        print(f'Processing SER for SNR = {snr} dB')
        errors = 0
        
        for i in range(sims):
            # Generate a random message
            message = np.random.randint(0, 2**sf)
            # Modulate the message
            signal = symbol_signals[message]
            # Generate AWGN
            noisy_signal = generate_awgn(f'{snr}dB', signal)[0]
            # Demodulate the signal
            demodulated_message = demod.demodulate_symbol(noisy_signal)
            # Calculate the number of errors
            if demodulated_message != message:
                errors += 1
        
        # Calculate SER for this SNR
        SER = errors / sims
        SNR_values.append(snr)
        SER_values.append(SER)
        print(f'SER for SNR = {snr} dB is {SER}')
    
    # Convert lists to numpy arrays
    SNR_values = np.array(SNR_values)
    SER_values = np.array(SER_values)

    # Save to a binary .npy file
    np.save(f'SER_SNR_ratio_sf{sf}.npy', np.vstack((SNR_values, SER_values)))

    print("SER-SNR Metrics successfully generated and saved in binary format for a Spreading Factor of", sf)

def plot_SER_SNR_from_binary(sf):
    # Load the binary file
    data = np.load(f'SER_SNR_ratio_sf{sf}.npy')
    SNR_values = data[0, :]
    SER_values = data[1, :]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_values, SER_values, marker='o', linestyle='-')
    plt.xlim([SNR_values.min(), SNR_values.max()])
    plt.yscale('log')  # Logarithmic scale for SER
    plt.xlabel('SNR (dB)')
    plt.ylabel('SER (log scale)')
    plt.title(f'SER vs. SNR for Spreading Factor {sf}')
    plt.grid(True)
    plt.show()

# Profile the generate_SER_SNR_ratio_binary function
cProfile.run('generate_SER_SNR_ratio_binary(7)', 'profile_output')

# Load the profiling results
p = pstats.Stats('profile_output')
p.sort_stats('cumulative').print_stats(10)

plot_SER_SNR_from_binary(7)