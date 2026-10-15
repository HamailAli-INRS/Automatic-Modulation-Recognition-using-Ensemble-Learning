"""
Isolated Impact Analysis: Demapping With Pilots vs Without Pilots.
Compares:
1. Classical Baseline WITH Pilots (Standard 5G NR DMRS, ~14.3% overhead)
2. Classical Demapper WITHOUT Pilots (Blind Demapping Failure)
3. Neural Demapper WITH Pilots (Hybrid Mode)
4. Neural Demapper WITHOUT Pilots (Zero-Pilot Full Spectral Efficiency)
"""

import time
import torch
import numpy as np
import copy

try:
    from ..models.rescnn_2d_demapper import ResCNNDemapper2D
    from ..pipeline.ofdm_channel_pipeline import OFDMChannelPipeline
    from ..pipeline.self_supervised_trainer import SelfSupervisedTrainer
    from ..benchmarks.classical_baseline import ClassicalBaseline
except Exception:
    from models.rescnn_2d_demapper import ResCNNDemapper2D
    from pipeline.ofdm_channel_pipeline import OFDMChannelPipeline
    from pipeline.self_supervised_trainer import SelfSupervisedTrainer
    from benchmarks.classical_baseline import ClassicalBaseline

class PilotAblationBenchmark:
    """
    Isolates the impact of pilot symbols on neural and classical demapping.
    """
    def __init__(self,
                 fft_size=64,
                 channel_type="rayleigh",
                 coderate=0.5,
                 device="cpu"):
        
        self.fft_size = fft_size
        self.channel_type = channel_type
        self.device = device
        
        # 1. Pilot-Assisted Pipeline
        self.pipe_pilots = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=fft_size,
            num_bits_per_symbol=4,
            coderate=coderate,
            channel_type=channel_type,
            pilot_mode="pilot_assisted",
            device=device
        )
        
        # 2. Pilot-Free Pipeline (Zero Pilots)
        self.pipe_zero_pilot = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=fft_size,
            num_bits_per_symbol=4,
            coderate=coderate,
            channel_type=channel_type,
            pilot_mode="pilot_free",
            device=device
        )
        
        self.classical_with_pilots = ClassicalBaseline(self.pipe_pilots)
        self.classical_no_pilots = ClassicalBaseline(self.pipe_zero_pilot)

    def run_pilot_ablation(self, train_steps=120, ebno_range=(8, 10, 12, 14, 16, 18, 20), eval_batches=32):
        print(f"\n[Pilot Ablation] Benchmarking With-Pilots vs Without-Pilots (Nsc={self.fft_size}, Channel={self.channel_type})...")
        
        # Train Neural Demapper (Zero Pilots)
        neural_zero_pilot = ResCNNDemapper2D(num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3).to(self.device)
        trainer_zero = SelfSupervisedTrainer(neural_zero_pilot, self.pipe_zero_pilot, device=self.device)
        print("  Training Neural Demapper in Zero-Pilot Mode...")
        trainer_zero.supervised_train(steps=train_steps, batch_size=16, verbose=False)
        
        # Train Neural Demapper (With Pilots)
        neural_with_pilots = ResCNNDemapper2D(num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3).to(self.device)
        trainer_pilots = SelfSupervisedTrainer(neural_with_pilots, self.pipe_pilots, device=self.device)
        print("  Training Neural Demapper in Pilot-Assisted Mode...")
        trainer_pilots.supervised_train(steps=train_steps, batch_size=16, verbose=False)
        
        results = {
            "ebno_range": list(ebno_range),
            "pilot_overhead_percent": self.pipe_pilots.pilot_overhead * 100.0,
            "classical_with_pilots_ber": [],
            "classical_with_pilots_bler": [],
            "classical_no_pilots_ber": [],
            "classical_no_pilots_bler": [],
            "neural_with_pilots_ber": [],
            "neural_with_pilots_bler": [],
            "neural_no_pilots_ber": [],
            "neural_no_pilots_bler": [],
            "spectral_efficiency_gain_percent": (1.0 / (1.0 - self.pipe_pilots.pilot_overhead) - 1.0) * 100.0
        }
        
        print("\n" + "=" * 96)
        print(f"{'Eb/N0':^8} | {'Classical (Pilots)':^20} | {'Classical (No Pilots)':^22} | {'Neural (Pilots)':^18} | {'Neural (Zero Pilots)':^20}")
        print("=" * 96)
        
        for ebno in ebno_range:
            _, c_fec, c_bler = self.classical_with_pilots.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["classical_with_pilots_ber"].append(c_fec)
            results["classical_with_pilots_bler"].append(c_bler)
            
            _, c_nopilot_fec, c_nopilot_bler = self.classical_no_pilots.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["classical_no_pilots_ber"].append(c_nopilot_fec)
            results["classical_no_pilots_bler"].append(c_nopilot_bler)
            
            _, n_pilot_fec, n_pilot_bler = trainer_pilots.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["neural_with_pilots_ber"].append(n_pilot_fec)
            results["neural_with_pilots_bler"].append(n_pilot_bler)
            
            _, n_nopilot_fec, n_nopilot_bler = trainer_zero.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["neural_no_pilots_ber"].append(n_nopilot_fec)
            results["neural_no_pilots_bler"].append(n_nopilot_bler)
            
            print(f" {ebno:4d} dB | {c_fec:20.5e} | {c_nopilot_fec:22.5e} | {n_pilot_fec:18.5e} | {n_nopilot_fec:20.5e}")
            
        print("=" * 96 + "\n")
        return results
