"""
Experience Replay Buffer for 64-QAM Online Self-Supervised Adaptation.
"""

import torch
import random

class ExperienceReplayBuffer64QAM:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.buffer = []

    def push(self, y_grid, c_target, is_valid_mask=None):
        batch_size = y_grid.shape[0]
        y_cpu = y_grid.detach().cpu()
        c_cpu = c_target.detach().cpu()
        
        if is_valid_mask is None:
            is_valid_mask = torch.ones(batch_size, dtype=torch.bool)
        else:
            is_valid_mask = is_valid_mask.detach().cpu()

        for i in range(batch_size):
            if bool(is_valid_mask[i]):
                if len(self.buffer) >= self.capacity:
                    self.buffer.pop(0)
                self.buffer.append((y_cpu[i], c_cpu[i]))

    def sample(self, batch_size, device="cpu"):
        sample_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, sample_size)
        y_batch = torch.stack([item[0] for item in batch], dim=0).to(device)
        c_batch = torch.stack([item[1] for item in batch], dim=0).to(device)
        return y_batch, c_batch

    def is_ready(self, min_size=8):
        return len(self.buffer) >= min_size

    def __len__(self):
        return len(self.buffer)
