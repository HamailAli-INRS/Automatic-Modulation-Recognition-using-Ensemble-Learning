"""
Self-Supervised Decision-Directed Online Trainer with Experience Replay Buffer.
Implements the continuous Delta-gradient feedback loop from the whiteboard diagram:
Distorted I/Q Grid -> Neural Demapper -> Raw LLRs -> LDPC Decoder (Spell-Checker) ->
Error-corrected Codeword -> Delta Loss -> Replay Buffer -> Backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class JointDemodFECTrainer:
    """
    Online Self-Supervised Trainer.
    
    Parameters:
    -----------
    demapper : nn.Module (ResCNNDemapper)
    pipeline : OFDMFECPipeline
    replay_buffer : ExperienceReplayBuffer
    lr : float
    weight_decay : float
    buffer_batch_size : int
    """
    def __init__(self,
                 demapper,
                 pipeline,
                 replay_buffer,
                 lr=1e-3,
                 weight_decay=1e-5,
                 buffer_batch_size=32,
                 device="cpu"):
        
        self.demapper = demapper.to(device)
        self.pipeline = pipeline
        self.replay_buffer = replay_buffer
        self.device = torch.device(device)
        self.buffer_batch_size = buffer_batch_size
        
        self.optimizer = optim.AdamW(self.demapper.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=1e-5)
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def bootstrap_warmup(self, steps=100, batch_size=16):
        """
        Phase 1: Multi-SNR geometric warmup so demapper learns 
        accurate LLR slicing across varying noise levels.
        """
        self.demapper.train()
        for step in range(1, steps + 1):
            snr = float(np.random.uniform(8.0, 18.0))
            u, c, y_grid, _ = self.pipeline.transmit(batch_size, ebno_db=snr)
            self.optimizer.zero_grad()
            llrs = self.demapper(y_grid)
            loss = self.criterion(llrs, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.demapper.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if step % max(1, steps // 5) == 0:
                pred_bits = (llrs > 0).float()
                raw_ber = (pred_bits != c).float().mean().item()
                print(f"  Warmup Step {step:3d}/{steps} | SNR: {snr:.1f} dB | Loss: {loss.item():.4f} | Raw BER: {raw_ber*100:.2f}%")

    def train_step(self, y_grid, ebno_db=None):
        """
        Phase 2: Execute online self-supervised adaptation step (Whiteboard Delta Loop).
        Zero labels or pilots used here! Only FEC-derived supervision.
        """
        self.demapper.train()
        batch_size = y_grid.shape[0]
        
        # Step 1: Forward pass through Neural Demapper to obtain raw LLRs
        raw_llrs = self.demapper(y_grid) # [B, N_coded_bits]
        
        # Step 2: Forward Error Correction (LDPC Belief Propagation - "Spell Checker")
        with torch.no_grad():
            u_hat, c_hat = self.pipeline.decode_ldpc(raw_llrs) # c_hat: [B, N_coded_bits]
            
            # Instantaneous raw hard bit decisions vs FEC corrected codeword
            pred_hard_bits = (raw_llrs > 0).float()
            raw_disagreement = (pred_hard_bits != c_hat).float().mean().item()
            
            # Reliability filter: accept frames into replay buffer
            frame_error_rates = (pred_hard_bits != c_hat).float().mean(dim=-1)
            valid_mask = frame_error_rates <= 0.30
            
        # Step 3: Compute Instantaneous Delta Loss against FEC-corrected pseudo-ground-truth
        instant_loss = self.criterion(raw_llrs, c_hat)
        
        # Step 4: Push to Experience Replay Buffer for stabilization
        self.replay_buffer.push(y_grid, c_hat, is_valid_mask=valid_mask)
        
        # Step 5: Sample from Replay Buffer and perform Gradient Descent Update
        self.optimizer.zero_grad()
        if self.replay_buffer.is_ready(min_size=8):
            buf_y, buf_c = self.replay_buffer.sample(self.buffer_batch_size, device=self.device)
            buf_llrs = self.demapper(buf_y)
            loss_delta = self.criterion(buf_llrs, buf_c)
            loss_delta.backward()
            
            torch.nn.utils.clip_grad_norm_(self.demapper.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            buf_loss_val = loss_delta.item()
        else:
            instant_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.demapper.parameters(), max_norm=1.0)
            self.optimizer.step()
            buf_loss_val = instant_loss.item()
            
        return {
            "instant_loss": instant_loss.item(),
            "buffer_loss": buf_loss_val,
            "raw_disagreement": raw_disagreement,
            "buffer_size": len(self.replay_buffer)
        }

    @torch.no_grad()
    def evaluate(self, batch_size=128, ebno_db=10.0):
        """
        Evaluate current model performance at given SNR.
        
        Returns:
        --------
        raw_ber : float (uncoded bit error rate from demapper)
        post_fec_ber : float (information bit error rate after LDPC decoding)
        bler : float (block/frame error rate)
        """
        self.demapper.eval()
        u_true, c_true, y_grid, _ = self.pipeline.transmit(batch_size, ebno_db)
        
        # Neural Demapper soft LLR estimation
        pred_llrs = self.demapper(y_grid)
        
        # Raw Demapper Hard Bit Decisions
        pred_c_bits = (pred_llrs > 0).float()
        raw_ber = (pred_c_bits != c_true).float().mean().item()
        
        # Post-FEC LDPC Decoding
        u_hat, _ = self.pipeline.decode_ldpc(pred_llrs)
        post_fec_ber = (u_hat != u_true).float().mean().item()
        
        # Block Error Rate (BLER)
        frame_errors = (u_hat != u_true).float().sum(dim=-1) > 0
        bler = frame_errors.float().mean().item()
        
        return raw_ber, post_fec_ber, bler
