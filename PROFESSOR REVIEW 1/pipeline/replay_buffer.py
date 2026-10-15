"""
Experience Replay Buffer for Online Self-Supervised Adaptation.
Stores validated parity-consistent OFDM frames to decouple gradient updates
from sequential channel correlations and prevent catastrophic forgetting.
"""

import random
import torch

class ExperienceReplayBuffer:
    """
    Fixed-capacity circular replay buffer storing (Distorted Grid, LDPC-Decoded Pseudo-Codeword) pairs.
    """
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, y_grid, c_hat, is_valid_mask=None):
        """
        Add frames and pseudo-codewords into the buffer.
        """
        y_grid_cpu = y_grid.detach().cpu()
        c_hat_cpu = c_hat.detach().cpu()
        
        batch_size = y_grid.shape[0]
        if is_valid_mask is None:
            is_valid_mask = torch.ones(batch_size, dtype=torch.bool)
        else:
            is_valid_mask = is_valid_mask.detach().cpu()

        for i in range(batch_size):
            if is_valid_mask[i].item():
                item = (y_grid_cpu[i], c_hat_cpu[i])
                if len(self.buffer) < self.capacity:
                    self.buffer.append(item)
                else:
                    self.buffer[self.position] = item
                self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device="cpu"):
        """Sample a batch uniformly at random from stored experiences."""
        actual_batch = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, actual_batch)
        y_batch = torch.stack([s[0] for s in samples]).to(device)
        c_batch = torch.stack([s[1] for s in samples]).to(device)
        return y_batch, c_batch

    def is_ready(self, min_size=16):
        return len(self.buffer) >= min_size

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.position = 0
