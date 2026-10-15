"""
Trainer for 64-QAM Neural LLR Corrector over 3GPP TDL Channel.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class JointTrainer64QAM:
    def __init__(self,
                 demapper,
                 pipeline,
                 lr=1e-3,
                 weight_decay=1e-5,
                 device="cpu"):
        
        self.demapper = demapper.to(device)
        self.pipeline = pipeline
        self.device = torch.device(device)
        
        self.optimizer = optim.AdamW(self.demapper.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def train_model(self, steps=150, batch_size=16):
        self.demapper.train()
        for step in range(1, steps + 1):
            snr = float(np.random.uniform(12.0, 22.0))
            u, c, x_hat, classical_llrs, _ = self.pipeline.transmit_and_receive(batch_size, ebno_db=snr)
            
            self.optimizer.zero_grad()
            refined_llrs = self.demapper(x_hat, classical_llrs)
            loss = self.criterion(refined_llrs, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.demapper.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            if step % max(1, steps // 5) == 0 or step == 1:
                pred_bits = (refined_llrs > 0).float()
                raw_ber = (pred_bits != c).float().mean().item()
                u_hat, _ = self.pipeline.decode_ldpc(refined_llrs)
                post_ber = (u_hat != u).float().mean().item()
                print(f"  Step {step:3d}/{steps} | SNR: {snr:.1f} dB | Loss: {loss.item():.4f} | Raw BER: {raw_ber*100:.2f}% | Post-FEC BER: {post_ber*100:.2f}%")

    @torch.no_grad()
    def evaluate(self, batch_size=64, ebno_db=16.0):
        self.demapper.eval()
        u_true, c_true, x_hat, classical_llrs, _ = self.pipeline.transmit_and_receive(batch_size, ebno_db)
        
        # Neural Demapper LLRs
        refined_llrs = self.demapper(x_hat, classical_llrs)
        pred_c_bits = (refined_llrs > 0).float()
        raw_ber = (pred_c_bits != c_true).float().mean().item()
        
        u_hat, _ = self.pipeline.decode_ldpc(refined_llrs)
        post_fec_ber = (u_hat != u_true).float().mean().item()
        bler = ((u_hat != u_true).float().sum(dim=-1) > 0).float().mean().item()
        
        return raw_ber, post_fec_ber, bler

    @torch.no_grad()
    def compute_ece_calibration(self, batch_size=128, ebno_db=18.0, num_bins=10):
        self.demapper.eval()
        _, c_true, x_hat, classical_llrs, _ = self.pipeline.transmit_and_receive(batch_size, ebno_db)
        llrs = self.demapper(x_hat, classical_llrs)
        
        probs = torch.sigmoid(llrs).contiguous().reshape(-1)
        labels = c_true.contiguous().reshape(-1)
        
        bin_edges = torch.linspace(0, 1, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        
        total_samples = len(labels)
        ece = 0.0
        
        for i in range(num_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i+1]
            mask = (probs >= bin_lower) & (probs < bin_upper) if i < num_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
            
            bin_count = mask.sum().item()
            if bin_count > 0:
                bin_acc = labels[mask].mean().item()
                bin_conf = probs[mask].mean().item()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
            else:
                bin_accuracies.append((bin_lower.item() + bin_upper.item()) / 2.0)
                bin_confidences.append((bin_lower.item() + bin_upper.item()) / 2.0)
                
        return {
            "ece": ece,
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies
        }
