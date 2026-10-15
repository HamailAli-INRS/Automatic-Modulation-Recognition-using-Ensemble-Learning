import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load('ofdm_dataset.npz')
rx_grids = data['rx_grids']
snrs     = data['snrs']
mods     = data['bits_per_symbol']

print('[OK] Loaded ofdm_dataset.npz:', rx_grids.shape, flush=True)

high_snr_idx = np.where(snrs > 12)[0][0]
sample_grid  = rx_grids[high_snr_idx]
sample_snr   = snrs[high_snr_idx]
sample_mod   = mods[high_snr_idx]

mod_names = {2: 'QPSK', 4: '16-QAM', 6: '64-QAM'}

plt.figure(figsize=(9, 5))
plt.imshow(np.abs(sample_grid).T, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Signal Magnitude')
plt.xlabel('OFDM Time Symbols (0 - 13)', fontsize=12)
plt.ylabel('Subcarrier Frequency Index (0 - 75)', fontsize=12)
plt.title('OFDM Frame Visualization (Magnitude Heatmap)', fontsize=13)
plt.tight_layout()
plt.savefig('dataset_grid_heatmap.png', dpi=150)
print('[OK] Saved OFDM Grid Image -> dataset_grid_heatmap.png', flush=True)

plt.figure(figsize=(6, 6))
plt.scatter(sample_grid.real.flatten(), sample_grid.imag.flatten(), color='blue', alpha=0.6, s=15)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('In-Phase (Real / I)', fontsize=12)
plt.ylabel('Quadrature (Imaginary / Q)', fontsize=12)
plt.title('I/Q Constellation Diagram', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('dataset_constellation.png', dpi=150)
print('[OK] Saved I/Q Constellation Image -> dataset_constellation.png', flush=True)
print('Done!', flush=True)
