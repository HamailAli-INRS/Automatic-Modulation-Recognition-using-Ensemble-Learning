import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Path to your dataset
filename = r'C:\Users\hamhamai001\Documents\AMR-Benchmark\data\RML2016.10a_dict.pkl'

print("Loading dataset... (this may take a moment, it's a large file)")
Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')

# Xd is a dictionary: keys are (modulation_type, SNR), values are arrays of signals
mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]

print(f"\nModulation types in dataset: {mods}")
print(f"SNR levels in dataset: {snrs}")
print(f"\nEach signal sample shape: {Xd[(mods[0], snrs[0])][0].shape}  <- (2 channels: I and Q, 128 time samples)")
print(f"Number of samples per (modulation, SNR) pair: {Xd[(mods[0], snrs[0])].shape[0]}")

# Pick a few modulation types and a high SNR (clean signal, easy to see) to visualize
chosen_mods = ['BPSK', 'QPSK', '8PSK', '16QAM', 'AM-DSB', 'WBFM']
chosen_snr = 18  # highest SNR = cleanest signal

fig, axes = plt.subplots(len(chosen_mods), 2, figsize=(12, 3 * len(chosen_mods)))
fig.suptitle(f'RML2016.10a Signal Samples at SNR = {chosen_snr} dB', fontsize=14)

for row, mod in enumerate(chosen_mods):
    if mod not in mods:
        continue
    sample = Xd[(mod, chosen_snr)][0]  # shape (2, 128): row 0 = I, row 1 = Q
    I = sample[0]
    Q = sample[1]

    # Left column: I/Q waveform over time
    axes[row, 0].plot(I, label='I (in-phase)', linewidth=1)
    axes[row, 0].plot(Q, label='Q (quadrature)', linewidth=1)
    axes[row, 0].set_title(f'{mod} — waveform')
    axes[row, 0].set_xlabel('Sample index')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].legend(fontsize=8)
    axes[row, 0].grid(alpha=0.3)

    # Right column: constellation diagram (I vs Q scatter)
    axes[row, 1].scatter(I, Q, s=8, alpha=0.6)
    axes[row, 1].set_title(f'{mod} — constellation (I vs Q)')
    axes[row, 1].set_xlabel('I')
    axes[row, 1].set_ylabel('Q')
    axes[row, 1].grid(alpha=0.3)
    axes[row, 1].axhline(0, color='gray', linewidth=0.5)
    axes[row, 1].axvline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('dataset_visualization.png', dpi=120)
print("\nSaved image to: dataset_visualization.png")
plt.show()