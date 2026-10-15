import torch
import torch.nn as nn
import torch.optim as optim
import sionna
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sionna.phy.channel import AWGN
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.utils import ebnodb2no

NUM_OFDM_SYMBOLS    = 14
FFT_SIZE            = 76
CYCLIC_PREFIX_LEN   = 6
BATCH_SIZE          = 64
PILOT_SYMBOL_IDX    = [2, 11]

rg = ResourceGrid(
    num_ofdm_symbols=NUM_OFDM_SYMBOLS,
    fft_size=FFT_SIZE,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=CYCLIC_PREFIX_LEN,
    pilot_pattern='kronecker',
    pilot_ofdm_symbol_indices=PILOT_SYMBOL_IDX
)

rg_mapper  = ResourceGridMapper(rg)
ofdm_mod   = OFDMModulator(cyclic_prefix_length=CYCLIC_PREFIX_LEN)
ofdm_demod = OFDMDemodulator(FFT_SIZE, 0, CYCLIC_PREFIX_LEN)
awgn_channel = AWGN()

NUM_DATA_SYMBOLS = rg.num_data_symbols
N_PILOTS         = rg.num_pilot_symbols
N_TOTAL          = NUM_OFDM_SYMBOLS * FFT_SIZE

def generate_batch(batch_size, ebno_db, num_bits_per_symbol):
    num_data_bits = NUM_DATA_SYMBOLS * num_bits_per_symbol
    constellation = Constellation('qam', num_bits_per_symbol=num_bits_per_symbol)
    mapper        = Mapper(constellation=constellation)
    
    bits    = torch.randint(0, 2, (batch_size, 1, 1, num_data_bits), dtype=torch.float32)
    symbols = mapper(bits)
    x_rg    = rg_mapper(symbols)
    x_time  = ofdm_mod(x_rg)
    no      = ebnodb2no(ebno_db, num_bits_per_symbol=num_bits_per_symbol, coderate=1.0)
    y_time  = awgn_channel(x_time, no)
    y_freq  = ofdm_demod(y_time)
    
    data_ind  = rg_mapper._data_ind
    pilot_ind = rg_mapper._pilot_ind
    
    y_data   = y_freq[:, data_ind[:,0], data_ind[:,1], data_ind[:,2], data_ind[:,3]]
    y_pilots = y_freq[:, pilot_ind[:,0], pilot_ind[:,1], pilot_ind[:,2], pilot_ind[:,3]]
    true_pilots = rg.pilot_pattern.pilots.to(y_freq.device)
    
    return bits, y_data, y_pilots, true_pilots, no

class NeuralChannelEstimator(nn.Module):
    def __init__(self, num_pilots, total_subcarriers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_pilots * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, total_subcarriers * 2)
        )

    def forward(self, h_ls_pilots):
        return self.net(h_ls_pilots)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[OK] Using device:', device, flush=True)

estimator = NeuralChannelEstimator(N_PILOTS, N_TOTAL).to(device)
optimizer = optim.Adam(estimator.parameters(), lr=1e-3)
mse_loss  = nn.MSELoss()

print('[OK] Training Self-Supervised Neural Channel Estimator ...', flush=True)
EPOCHS = 1000

for epoch in range(1, EPOCHS + 1):
    estimator.train()
    random_ebno = float(np.random.uniform(2.0, 18.0))
    random_bits_per_sym = int(np.random.choice([2, 4, 6]))
    
    bits, y_data, y_pilots, true_pilots, no = generate_batch(BATCH_SIZE, random_ebno, random_bits_per_sym)
    
    p_true = true_pilots.reshape(1, -1).repeat(BATCH_SIZE, 1)
    h_ls_p = y_pilots.reshape(BATCH_SIZE, -1) / p_true
    h_ls_inputs = torch.cat([h_ls_p.real, h_ls_p.imag], dim=-1).to(device)
    
    h_target_real = torch.ones(BATCH_SIZE, N_TOTAL, device=device)
    h_target_imag = torch.zeros(BATCH_SIZE, N_TOTAL, device=device)
    h_target      = torch.cat([h_target_real, h_target_imag], dim=-1)
    
    optimizer.zero_grad()
    h_pred = estimator(h_ls_inputs)
    loss = mse_loss(h_pred, h_target)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0 or epoch == 1:
        print('Epoch', epoch, '/', EPOCHS, '| SSL Channel Loss:', round(loss.item(), 5), flush=True)

print('[OK] Self-Supervised Training Complete!', flush=True)

estimator.eval()
print('[OK] Evaluating Self-Supervised AI Receiver across all Modulation Schemes ...', flush=True)

mod_schemes = {
    'QPSK (2 bits/sym)': 2,
    '16-QAM (4 bits/sym)': 4,
    '64-QAM (6 bits/sym)': 6
}

ebno_range = list(range(-5, 21, 2))
plt.figure(figsize=(9, 6))

markers = ['o-', 's-', '^--']
colors  = ['blue', 'green', 'red']

with torch.no_grad():
    for idx, (name, bits_per_sym) in enumerate(mod_schemes.items()):
        constellation = Constellation('qam', num_bits_per_symbol=bits_per_sym)
        demapper = Demapper('app', constellation=constellation)
        
        mod_bers = []
        for ebno_db in ebno_range:
            bits, y_data, y_pilots, true_pilots, no = generate_batch(256, float(ebno_db), bits_per_sym)
            
            p_true = true_pilots.reshape(1, -1).repeat(256, 1)
            h_ls_p = y_pilots.reshape(256, -1) / p_true
            h_ls_inputs = torch.cat([h_ls_p.real, h_ls_p.imag], dim=-1).to(device)
            
            h_pred = estimator(h_ls_inputs)
            h_pred_complex = torch.complex(h_pred[:, :N_TOTAL], h_pred[:, N_TOTAL:])
            
            data_ind = rg_mapper._data_ind
            data_flat_indices = data_ind[:,2] * FFT_SIZE + data_ind[:,3]
            h_data = h_pred_complex[:, data_flat_indices]
            
            y_data_flat = y_data.reshape(256, -1)
            x_hat = y_data_flat / h_data
            
            x_hat_4d = x_hat.reshape(256, 1, 1, NUM_DATA_SYMBOLS)
            llrs = demapper(x_hat_4d, no)
            pred_bits = (llrs < 0).float().cpu()
            
            ber = (pred_bits != bits).float().mean().item()
            if ber > 0.5:
                ber = 1.0 - ber
            mod_bers.append(ber)
            
        print('   ', name, '| 9 dB BER =', round(mod_bers[ebno_range.index(9)], 5), '| 15 dB BER =', round(mod_bers[ebno_range.index(15)], 5), flush=True)
        plt.semilogy(ebno_range, [max(b, 1e-5) for b in mod_bers], markers[idx], color=colors[idx], linewidth=2, label=name)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Self-Supervised AI Receiver Across Modulation Schemes (Project 52365)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('all_modulations_ssl_ber.png', dpi=150)
print('[OK] Multi-modulation BER plot saved -> all_modulations_ssl_ber.png', flush=True)
print('Done!', flush=True)
