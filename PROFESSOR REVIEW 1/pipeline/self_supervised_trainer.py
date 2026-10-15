"""
Self-Supervised Decision-Directed Online Trainer & Multi-Mode Evaluator.
Implements the continuous Delta-gradient feedback loop:
Distorted I/Q Grid -> 2D ResCNN Demapper -> Raw LLRs -> LDPC Decoder (Spell-Checker) ->
Error-corrected Codeword -> Delta Loss -> Replay Buffer -> Backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

try:
    from .replay_buffer import ExperienceReplayBuffer
except Exception:
    from pipeline.replay_buffer import ExperienceReplayBuffer

class SelfSupervisedTrainer:
    """
    Self-Supervised Online Trainer with Experience Replay Buffer and Multi-Mode Benchmark Support.
    
    Parameters:
    -----------
    demapper : nn.Module (ResCNNDemapper2D)
    pipeline : OFDMChannelPipeline
    replay_buffer : ExperienceReplayBuffer (optional)
    lr : float
    weight_decay : float
    buffer_batch_size : int
    device : str
    """
    def __init__(self,
                 demapper,
                 pipeline,
                 replay_buffer=None,
                 lr=1e-3,
                 weight_decay=1e-5,
                 buffer_batch_size=32,
                 device="cpu"):
        
        self.demapper = demapper.to(device)
        self.pipeline = pipeline
        self.replay_buffer = replay_buffer if replay_buffer is not None else ExperienceReplayBuffer()
        self.device = torch.device(device)
        self.buffer_batch_size = buffer_batch_size
        
        self.optimizer = optim.AdamW(self.demapper.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def supervised_train(self, steps=100, batch_size=16, snr_min=8.0, snr_max=18.0, verbose=True):
        """
        Supervised training phase (using true encoded bits 'c') as an upper-bound reference or warmup.
        """
        self.demapper.train()
        losses = []
        for step in range(1, steps + 1):
            snr = float(np.random.uniform(snr_min, snr_max))
            u, c, y_grid, _, _ = self.pipeline.transmit(batch_size, ebno_db=snr)
            
            self.optimizer.zero_grad()
            llrs = self.demapper(y_grid, data_mask=self.pipeline.data_mask)
            loss = self.criterion(llrs, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.demapper.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            losses.append(loss.item())
            
            if verbose and step % max(1, steps // 5) == 0:
                pred_bits = (llrs > 0).float()
                raw_ber = (pred_bits != c).float().mean().item()
                print(f"  Supervised Step {step:3d}/{steps} | SNR: {snr:4.1f} dB | Loss: {loss.item():.4f} | Raw BER: {raw_ber*100:5.2f}%")
        return float(np.mean(losses[-10:]))

    def self_supervised_delta_step(self, y_grid):
        """
        Execute one online self-supervised adaptation step (Delta Loop).
        100% Blind / Decision-Directed: Zero ground-truth bits or pilots used!
        """
        self.demapper.train()
        
        # 1. Forward pass through Neural Demapper to estimate soft LLRs
        raw_llrs = self.demapper(y_grid, data_mask=self.pipeline.data_mask)
        
        # 2. Forward Error Correction (5G NR LDPC Belief Propagation "Spell Checker")
        with torch.no_grad():
            u_hat, c_hat = self.pipeline.decode_ldpc(raw_llrs)
            pred_hard_bits = (raw_llrs > 0).float()
            raw_disagreement = (pred_hard_bits != c_hat).float().mean().item()
            
            # Confidence filter: only admit blocks whose raw error is within LDPC error correction capability
            frame_error_rates = (pred_hard_bits != c_hat).float().mean(dim=-1)
            valid_mask = frame_error_rates <= 0.35
            
        # 3. Compute Delta Loss against LDPC-derived pseudo-codeword
        instant_loss = self.criterion(raw_llrs, c_hat)
        
        # 4. Push to Experience Replay Buffer
        self.replay_buffer.push(y_grid, c_hat, is_valid_mask=valid_mask)
        
        # 5. Gradient Descent update from Replay Buffer
        self.optimizer.zero_grad()
        if self.replay_buffer.is_ready(min_size=8):
            buf_y, buf_c = self.replay_buffer.sample(self.buffer_batch_size, device=self.device)
            buf_llrs = self.demapper(buf_y, data_mask=self.pipeline.data_mask)
            loss_delta = self.criterion(buf_llrs, buf_c)
            loss_delta.backward()
            torch.nn.utils.clip_grad_norm_(self.demapper.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            effective_loss = loss_delta.item()
        else:
            instant_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.demapper.parameters(), max_norm=1.0)
            self.optimizer.step()
            effective_loss = instant_loss.item()
            
        return {
            "loss": effective_loss,
            "disagreement": raw_disagreement,
            "buffer_size": len(self.replay_buffer)
        }

    def online_self_supervised_adapt(self, stream_steps=50, batch_size=8, snr_db=14.0):
        """
        Simulate an online data stream undergoing self-supervised adaptation.
        """
        for step in range(stream_steps):
            _, _, y_grid, _, _ = self.pipeline.transmit(batch_size, ebno_db=snr_db)
            self.self_supervised_delta_step(y_grid)

    @torch.no_grad()
    def evaluate(self, batch_size=64, ebno_db=10.0):
        """
        Evaluate neural demapper BER, Post-FEC BER, and BLER at target SNR.
        """
        self.demapper.eval()
        u_true, c_true, y_grid, _, _ = self.pipeline.transmit(batch_size, ebno_db)
        
        pred_llrs = self.demapper(y_grid, data_mask=self.pipeline.data_mask)
        pred_c_bits = (pred_llrs > 0).float()
        raw_ber = (pred_c_bits != c_true).float().mean().item()
        
        u_hat, _ = self.pipeline.decode_ldpc(pred_llrs)
        post_fec_ber = (u_hat != u_true).float().mean().item()
        
        frame_errors = (u_hat != u_true).float().sum(dim=-1) > 0
        bler = frame_errors.float().mean().item()
        
        return raw_ber, post_fec_ber, bler

    @torch.no_grad()
    def compute_ece_calibration(self, batch_size=64, ebno_db=14.0, num_bins=10):
        """
        Calculate Expected Calibration Error (ECE) for LLR outputs.
        """
        self.demapper.eval()
        _, c_true, y_grid, _, _ = self.pipeline.transmit(batch_size, ebno_db)
        logits = self.demapper(y_grid, data_mask=self.pipeline.data_mask)
        
        probs = torch.sigmoid(logits).view(-1)
        labels = c_true.view(-1)
        
        confidences = torch.where(probs >= 0.5, probs, 1.0 - probs)
        predictions = (probs >= 0.5).float()
        accuracies = (predictions == labels).float()
        
        bin_boundaries = torch.linspace(0.5, 1.0, num_bins + 1)
        ece = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper if i < num_bins - 1 else confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()
            
            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].mean().item()
                conf_in_bin = confidences[in_bin].mean().item()
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
                bin_accs.append(acc_in_bin)
                bin_confs.append(conf_in_bin)
                bin_counts.append(int(in_bin.sum().item()))
            else:
                bin_accs.append(0.0)
                bin_confs.append((bin_lower.item() + bin_upper.item()) / 2.0)
                bin_counts.append(0)
                
        return {
            "ece": float(ece),
            "bin_accs": bin_accs,
            "bin_confs": bin_confs,
            "bin_counts": bin_counts
        }
