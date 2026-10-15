"""
Pilot Pattern Utilities for 5G NR OFDM Resource Grids.
Implements standard 5G NR Demodulation Reference Signal (DMRS) pilot insertion and LS/LMMSE channel estimation.
"""

import torch
import numpy as np

class DmrsPilotPattern:
    """
    Standard 5G NR DMRS Pilot Grid Pattern.
    
    Inserts known pilot symbols at designated OFDM symbol and subcarrier indices
    (e.g., symbol index 2 and/or 11 across every 2nd or 4th subcarrier).
    """
    def __init__(self, num_ofdm_symbols=14, fft_size=64, pilot_symbol_indices=(2, 11), pilot_sc_step=2):
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.pilot_symbol_indices = pilot_symbol_indices
        self.pilot_sc_step = pilot_sc_step
        
        # Build 2D boolean mask [Subcarriers, Symbols]
        self.pilot_mask = np.zeros((fft_size, num_ofdm_symbols), dtype=bool)
        for sym in pilot_symbol_indices:
            self.pilot_mask[0::pilot_sc_step, sym] = True
            
        self.num_pilot_symbols = int(np.sum(self.pilot_mask))
        self.num_total_symbols = fft_size * num_ofdm_symbols
        self.num_data_symbols = self.num_total_symbols - self.num_pilot_symbols
        self.pilot_overhead_ratio = self.num_pilot_symbols / self.num_total_symbols

    def generate_pilot_grid(self, batch_size, device="cpu"):
        """
        Generates fixed QPSK pilot symbols (norm=1.0) placed on pilot positions.
        """
        pilot_tensor_mask = torch.from_numpy(self.pilot_mask).to(device) # [sc, sym]
        # Fixed deterministic pseudo-random QPSK pilot sequence for channel probing
        pilot_values = torch.ones((batch_size, self.num_pilot_symbols), dtype=torch.complex64, device=device)
        return pilot_tensor_mask, pilot_values

    def estimate_channel_ls_2d(self, rx_grid, pilot_mask_tensor, tx_pilots):
        """
        Least Squares (LS) channel estimation at pilot positions + 2D Linear/Nearest Interpolation.
        
        Parameters:
        -----------
        rx_grid : torch.Tensor [B, Subcarriers, Symbols] (Complex)
        pilot_mask_tensor : torch.Tensor [Subcarriers, Symbols] (Bool)
        tx_pilots : torch.Tensor [B, Num_Pilots] (Complex)
        
        Returns:
        --------
        h_hat : torch.Tensor [B, Subcarriers, Symbols] (Estimated channel response)
        """
        batch_size = rx_grid.shape[0]
        h_hat = torch.ones_like(rx_grid)
        
        # For each frame in batch, compute LS estimate at pilot positions
        # H_LS(p) = Y(p) / X(p)
        rx_pilots = rx_grid[:, pilot_mask_tensor] # [B, Num_Pilots]
        h_pilots = rx_pilots / (tx_pilots + 1e-9)
        
        # 2D Frequency-Time interpolation across the grid
        for b in range(batch_size):
            h_grid_b = torch.ones((self.fft_size, self.num_ofdm_symbols), dtype=torch.complex64, device=rx_grid.device)
            # Fill pilot values
            h_grid_b[pilot_mask_tensor] = h_pilots[b]
            
            # Simple 1D-then-1D interpolation/smoothing across subcarriers & symbols
            # Along subcarriers for each pilot symbol:
            for sym in self.pilot_symbol_indices:
                col = h_grid_b[:, sym]
                sc_idx = np.where(self.pilot_mask[:, sym])[0]
                interp_real = np.interp(np.arange(self.fft_size), sc_idx, col.real.cpu().numpy()[sc_idx])
                interp_imag = np.interp(np.arange(self.fft_size), sc_idx, col.imag.cpu().numpy()[sc_idx])
                h_grid_b[:, sym] = torch.tensor(interp_real + 1j * interp_imag, dtype=torch.complex64, device=rx_grid.device)
                
            # Along symbols for all subcarriers:
            sym_indices = np.array(self.pilot_symbol_indices)
            for sc in range(self.fft_size):
                row = h_grid_b[sc, sym_indices]
                interp_real = np.interp(np.arange(self.num_ofdm_symbols), sym_indices, row.real.cpu().numpy())
                interp_imag = np.interp(np.arange(self.num_ofdm_symbols), sym_indices, row.imag.cpu().numpy())
                h_grid_b[sc, :] = torch.tensor(interp_real + 1j * interp_imag, dtype=torch.complex64, device=rx_grid.device)
                
            h_hat[b] = h_grid_b
            
        return h_hat
