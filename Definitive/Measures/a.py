import os
import numpy as np


def unify():
    # Load the binary file
    snr_range = [-5,-4,-3,-2,-1]
    BER_values_og = []
    BER_values_corr = []
    for snr in snr_range:
        filename_og = f'Measures/FramePerf/BER_SNR{snr}_sf7_bw125k_spc1_og_sync.npy'
        filename_corr = f'Measures/FramePerf/BER_SNR{snr}_sf7_bw125k_spc1_corr_sync.npy'
        og_data = np.load(filename_og)
        corr_data = np.load(filename_corr)
        BER_values_og.append(og_data[1])
        BER_values_corr.append(corr_data[1])
    print(BER_values_og)
    print(BER_values_corr)
    snr_range = np.array(snr_range)
    BER_values_og = np.array(BER_values_og).flatten()
    BER_values_corr = np.array(BER_values_corr).flatten()
    # Save the unified data
    np.save('Measures\FramePerf\BER_og.npy', np.vstack((snr_range, BER_values_og)))
    np.save('Measures\FramePerf\BER_corr.npy', np.vstack((snr_range, BER_values_corr)))
    # Plot the data

unify()
from matplotlib import pyplot as plt
def plot_SER_SNR_from_binary(filename, sf ,spc, label_fill):
    # Load the binary file
    data = np.load(filename)
    SNR_values = data[0, :]
    BER_values = data[1, :]
    # Plot the data
    plt.plot(SNR_values, BER_values, marker='o', linestyle='-', label=label_fill)
    plt.xlim([SNR_values.min(), SNR_values.max()])
    plt.yscale('log')  # Logarithmic scale for BER
    plt.xticks(np.arange(-30 + 12 - sf, 2, 1))
    plt.xlabel('SNR (dB)')
    plt.ylabel('SER')
    
    plt.grid(True)