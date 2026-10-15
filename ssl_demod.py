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
NUM_BITS_PER_SYMBOL = 4
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

constellation = Constellation('qam', num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
mapper        = Mapper(constellation=constellation)
rg_mapper     = ResourceGridMapper(rg)
ofdm_mod      = OFDMModulator(cyclic_prefix_length=CYCLIC_PREFIX_LEN)
ofdm_demod    = OFDMDemodulator(FFT_SIZE, 0, CYCLIC_PREFIX_LEN)
awgn_channel  = AWGN()

NUM_DATA_SYMBOLS  = rg.num_data_symbols
NUM_PILOT_SYMBOLS = rg.num_pilot_symbols
NUM_DATA_BITS     = NUM_DATA_SYMBOLS * NUM_BITS_PER_SYMBOL

def generate_batch(batch_size, ebno_db):
    bits    = torch.randint(0, 2, (batch_size, 1, 1, NUM_DATA_BITS), dtype=torch.float32)
    symbols = mapper(bits)
    x_rg    = rg_mapper(symbols)
    x_time  = ofdm_mod(x_rg)
    no      = ebnodb2no(ebno_db, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=1.0)
    y_time  = awgn_channel(x_time, no)
    y_freq  = ofdm_demod(y_time)
    
    data_ind = rg_mapper._data_ind
    pilot_ind = rg_mapper._pilot_ind
    
    y_data   = y_freq[:, data_ind[:,0], data_ind[:,1], data_ind[:,2], data_ind[:,3]]
    y_pilots = y_freq[:, pilot_ind[:,0], pilot_ind[:,1], pilot_ind[:,2], pilot_ind[:,3]]
    
    true_pilots = rg.pilot_pattern.pilots.to(y_freq.device)
    return bits, y_data, y_pilots, true_pilots

class SSLDemodulator(nn.Module):
    def __init__(self, num_bits_per_symbol):
        super().__init__()
        self.demapper = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_bits_per_symbol)
        )

    def forward(self, y_subcarriers):
        return self.demapper(y_subcarriers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[OK] Using device:', device, flush=True)

ssl_model = SSLDemodulator(NUM_BITS_PER_SYMBOL).to(device)
optimizer = optim.Adam(ssl_model.parameters(), lr=1e-3)
mse_loss  = nn.MSELoss()

print('[OK] Starting Pilot-Assisted Self-Supervised Training (NO BIT LABELS USED) ...', flush=True)
EPOCHS = 1000

for epoch in range(1, EPOCHS + 1):
    ssl_model.train()
    random_ebno = float(np.random.uniform(4.0, 18.0))
    bits, y_data, y_pilots, true_pilots = generate_batch(BATCH_SIZE, random_ebno)
    
    # Received pilots per batch: [BATCH_SIZE * NUM_PILOTS, 2]
    y_p_flat = y_pilots.reshape(-1)
    pilot_inputs = torch.stack([y_p_flat.real, y_p_flat.imag], dim=-1).to(device)
    
    # Ground truth pilots broadcasted across batch
    p_true_flat = true_pilots.reshape(-1)
    p_true_vec  = torch.stack([p_true_flat.real, p_true_flat.imag], dim=-1).to(device).repeat(BATCH_SIZE, 1)
    
    optimizer.zero_grad()
    pilot_logits = ssl_model(pilot_inputs)
    
    # Soft symbol reconstruction from bit logits
    bit_probs = torch.sigmoid(pilot_logits)
    reconstructed_real = (2 * bit_probs[:, 0] - 1) + 0.5 * (2 * bit_probs[:, 1] - 1)
    reconstructed_imag = (2 * bit_probs[:, 2] - 1) + 0.5 * (2 * bit_probs[:, 3] - 1)
    reconstructed_symbols = torch.stack([reconstructed_real, reconstructed_imag], dim=-1)
    
    # SELF-SUPERVISED LOSS: evaluated ONLY on pilot subcarriers!
    loss = mse_loss(reconstructed_symbols, p_true_vec)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0 or epoch == 1:
        print('Epoch', epoch, '/', EPOCHS, '| Pilot SSL Loss:', round(loss.item(), 5), '| SNR:', round(random_ebno, 1), 'dB', flush=True)

print('[OK] Self-Supervised Training complete!', flush=True)

ssl_model.eval()
print('[OK] Evaluating Self-Supervised AI Demodulator on DATA subcarriers ...', flush=True)

ebno_range = list(range(-5, 21, 2))
ssl_bers = []

with torch.no_grad():
    for ebno_db in ebno_range:
        bits, y_data, _, _ = generate_batch(256, float(ebno_db))
        true_bits = bits.reshape(-1, NUM_BITS_PER_SYMBOL)
        
        y_d_flat = y_data.reshape(-1)
        data_inputs = torch.stack([y_d_flat.real, y_d_flat.imag], dim=-1).to(device)
        
        logits = ssl_model(data_inputs)
        pred_bits = (torch.sigmoid(logits) > 0.5).float().cpu()
        
        ber = (pred_bits != true_bits).float().mean().item()
        ssl_bers.append(ber)
        print('   Eb/N0 =', ebno_db, 'dB | Pilot-SSL AI Demod BER =', round(ber, 5), flush=True)

plt.figure(figsize=(8, 5))
plt.semilogy(ebno_range, [max(b, 1e-5) for b in ssl_bers], 'g-^', linewidth=2, label='Self-Supervised AI (Pilot-Assisted Loss)')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Self-Supervised AI Demodulator (Project ID 52365)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ssl_demod_ber.png', dpi=150)
print('[OK] SSL BER plot saved -> ssl_demod_ber.png', flush=True)
print('Done!', flush=True)
