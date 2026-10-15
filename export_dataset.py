import torch
import sionna
import numpy as np

from sionna.phy.channel import AWGN
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.utils import ebnodb2no

NUM_FRAMES          = 500
NUM_OFDM_SYMBOLS    = 14
FFT_SIZE            = 76
CYCLIC_PREFIX_LEN   = 6
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

rg_mapper    = ResourceGridMapper(rg)
ofdm_mod     = OFDMModulator(cyclic_prefix_length=CYCLIC_PREFIX_LEN)
ofdm_demod   = OFDMDemodulator(FFT_SIZE, 0, CYCLIC_PREFIX_LEN)
awgn_channel = AWGN()

NUM_DATA_SYMBOLS = rg.num_data_symbols

def generate_and_save_dataset(filename='ofdm_dataset.npz'):
    print('[OK] Generating OFDM dataset samples ...', flush=True)
    
    rx_grids = []
    tx_grids = []
    snrs     = []
    bits_per_sym_list = []
    
    for i in range(NUM_FRAMES):
        bits_per_sym = int(np.random.choice([2, 4, 6]))
        ebno_db      = float(np.random.uniform(-5.0, 20.0))
        
        num_data_bits = NUM_DATA_SYMBOLS * bits_per_sym
        constellation = Constellation('qam', num_bits_per_symbol=bits_per_sym)
        mapper        = Mapper(constellation=constellation)
        
        bits    = torch.randint(0, 2, (1, 1, 1, num_data_bits), dtype=torch.float32)
        symbols = mapper(bits)
        x_rg    = rg_mapper(symbols)
        x_time  = ofdm_mod(x_rg)
        no      = ebnodb2no(ebno_db, num_bits_per_symbol=bits_per_sym, coderate=1.0)
        y_time  = awgn_channel(x_time, no)
        y_freq  = ofdm_demod(y_time)
        
        tx_grids.append(x_rg.numpy().squeeze())
        rx_grids.append(y_freq.numpy().squeeze())
        snrs.append(ebno_db)
        bits_per_sym_list.append(bits_per_sym)
    
    np.savez_compressed(
        filename,
        rx_grids=np.array(rx_grids),
        tx_grids=np.array(tx_grids),
        snrs=np.array(snrs),
        bits_per_symbol=np.array(bits_per_sym_list)
    )
    
    print('[OK] Dataset saved successfully ->', filename, flush=True)
    print('     Saved', NUM_FRAMES, 'OFDM frames [Rx shape:', np.array(rx_grids).shape, ']', flush=True)

if __name__ == '__main__':
    generate_and_save_dataset()
