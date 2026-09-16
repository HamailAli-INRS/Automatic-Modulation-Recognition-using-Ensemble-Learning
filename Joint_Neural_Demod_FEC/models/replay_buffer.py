"""
Experience Replay Buffer for Self-Supervised Decision-Directed Online Learning.
Buffers received I/Q resource grids and verified error-corrected codewords to calculate
stable mini-batch loss gradients (Delta loop) without catastrophic forgetting.
"""

import torch
import random

class ExperienceReplayBuffer:
    """
    Circular Experience Replay Buffer.
    
    Parameters:
    -----------
    capacity : int
        Maximum number of OFDM frames stored in memory.
    """
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, y_grid, corrected_codeword, is_valid_mask=None):
        """
        Store a batch of frames into the buffer.
        
        Parameters:
        -----------
        y_grid : torch.Tensor
            Received distorted I/Q grids [Batch, 2, Subcarriers, Symbols] or complex [Batch, Subcarriers, Symbols].
        corrected_codeword : torch.Tensor
            Error-corrected binary codewords [Batch, Codeword_Length].
        is_valid_mask : torch.Tensor or None
            Boolean tensor indicating whether each frame passed parity/syndrome check.
        """
        if torch.is_complex(y_grid):
            y_grid = torch.stack([y_grid.real, y_grid.imag], dim=1).float()
            
        batch_size = y_grid.shape[0]
        y_grid_cpu = y_grid.detach().cpu()
        codeword_cpu = corrected_codeword.detach().cpu()
        
        for i in range(batch_size):
            if is_valid_mask is not None and not is_valid_mask[i].item():
                continue # Skip corrupt frames to prevent gradient noise
                
            sample = (y_grid_cpu[i], codeword_cpu[i])
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
                
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device="cpu"):
        """
        Sample a mini-batch of stored experiences.
        
        Returns:
        --------
        y_grids : torch.Tensor [B, 2, Subcarriers, Symbols]
        codewords : torch.Tensor [B, Codeword_Length]
        """
        if len(self.buffer) == 0:
            return None, None
            
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)
        
        y_grids = torch.stack([item[0] for item in batch]).to(device)
        codewords = torch.stack([item[1] for item in batch]).to(device)
        return y_grids, codewords

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, min_size=16):
        return len(self.buffer) >= min_size
