"""
Self-Supervised Delta Loop Ablation Benchmark.
Quantifies the exact gain provided by the continuous decision-directed LDPC Delta Feedback Loop:
1. Frozen / Non-Adapted Demapper (without Delta Loop)
2. Online Self-Supervised Delta-Adapted Demapper (with Delta Loop + Replay Buffer)
3. Fully Supervised Demapper (Upper Bound)
4. Classical Baseline
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

class SelfSupervisedDeltaBenchmark:
    """
    Evaluates the performance difference between Self-Supervised Delta Loop ON vs OFF.
    """
    def __init__(self,
                 fft_size=64,
                 channel_type="tdl_raytracing",
                 coderate=0.5,
                 device="cpu"):
        
        self.fft_size = fft_size
        self.channel_type = channel_type
        self.device = device
        
        # Pipeline in Pilot-Free Mode (Zero Pilots)
        self.pipeline = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=fft_size,
            num_bits_per_symbol=4,
            coderate=coderate,
            channel_type=channel_type,
            pilot_mode="pilot_free",
            device=device
        )
        
        # Classical Baseline pipeline with pilots for comparison
        self.pilot_pipeline = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=fft_size,
            num_bits_per_symbol=4,
            coderate=coderate,
            channel_type=channel_type,
            pilot_mode="pilot_assisted",
            device=device
        )
        self.classical_baseline = ClassicalBaseline(self.pilot_pipeline)

    def run_ablation(self, warmup_steps=120, adapt_stream_steps=80, ebno_range=(8, 10, 12, 14, 16, 18, 20), eval_batches=32):
        """
        Executes the comparison across Eb/N0 range.
        """
        print(f"\n[Delta-Loop Ablation] Initializing models (Nsc={self.fft_size}, Info Bits={self.pipeline.k})...")
        
        # Base Model
        base_model = ResCNNDemapper2D(num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3).to(self.device)
        base_trainer = SelfSupervisedTrainer(base_model, self.pipeline, device=self.device)
        
        # Initial Warmup / Bootstrap
        print("  Bootstrapping initial Demapper weights...")
        base_trainer.supervised_train(steps=warmup_steps, batch_size=16, verbose=False)
        
        # Create Frozen (No Adaptation) and Delta-Adapted copies
        frozen_model = copy.deepcopy(base_model)
        frozen_trainer = SelfSupervisedTrainer(frozen_model, self.pipeline, device=self.device)
        
        delta_model = copy.deepcopy(base_model)
        delta_trainer = SelfSupervisedTrainer(delta_model, self.pipeline, device=self.device)
        
        # Supervised Upper Bound Model
        supervised_model = copy.deepcopy(base_model)
        supervised_trainer = SelfSupervisedTrainer(supervised_model, self.pipeline, device=self.device)
        print("  Training Supervised Reference (Upper Bound)...")
        supervised_trainer.supervised_train(steps=warmup_steps + adapt_stream_steps, batch_size=16, verbose=False)
        
        # Online adaptation for Delta model
        print("  Running Online Self-Supervised Delta-Loop Adaptation...")
        delta_trainer.online_self_supervised_adapt(stream_steps=adapt_stream_steps, batch_size=16, snr_db=14.0)
        
        results = {
            "ebno_range": list(ebno_range),
            "frozen_raw_ber": [],
            "frozen_post_fec_ber": [],
            "frozen_bler": [],
            "delta_raw_ber": [],
            "delta_post_fec_ber": [],
            "delta_bler": [],
            "supervised_raw_ber": [],
            "supervised_post_fec_ber": [],
            "supervised_bler": [],
            "classical_pilot_fec_ber": [],
            "classical_pilot_bler": []
        }
        
        print("\n" + "=" * 88)
        print(f"{'Eb/N0 (dB)':^10} | {'Frozen (No Delta)':^18} | {'Self-Supervised (Delta)':^24} | {'Supervised (UB)':^18} | {'Delta BLER':^10}")
        print("=" * 88)
        
        for ebno in ebno_range:
            # Frozen
            f_raw, f_fec, f_bler = frozen_trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["frozen_raw_ber"].append(f_raw)
            results["frozen_post_fec_ber"].append(f_fec)
            results["frozen_bler"].append(f_bler)
            
            # Delta-adapted
            d_raw, d_fec, d_bler = delta_trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["delta_raw_ber"].append(d_raw)
            results["delta_post_fec_ber"].append(d_fec)
            results["delta_bler"].append(d_bler)
            
            # Supervised
            s_raw, s_fec, s_bler = supervised_trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["supervised_raw_ber"].append(s_raw)
            results["supervised_post_fec_ber"].append(s_fec)
            results["supervised_bler"].append(s_bler)
            
            # Classical with pilots
            _, c_fec, c_bler = self.classical_baseline.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
            results["classical_pilot_fec_ber"].append(c_fec)
            results["classical_pilot_bler"].append(c_bler)
            
            print(f" {ebno:4d} dB   | {f_fec:18.5e} | {d_fec:24.5e} | {s_fec:18.5e} | {d_bler:10.4f}")
            
        print("=" * 88 + "\n")
        return results
