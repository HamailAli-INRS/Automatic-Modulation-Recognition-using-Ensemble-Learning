"""
Master Orchestration Script for Professor Review 1 Experiments.
Executes end-to-end evaluations addressing all 5 items from the Professor Meeting:
1. Scaled Information Bits (k=1792, k=3584, k=7168)
2. Realistic Multipath Dynamics (AWGN vs Rayleigh vs 3GPP TDL Ray-Tracing)
3. Self-Supervised vs Non-Self-Supervised (Delta Feedback Loop quantification)
4. Isolated Neural Demapper Impact (With Pilots vs Zero Pilots)
5. Computational Complexity & Latency Profiling
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np

# Ensure local project path is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.rescnn_2d_demapper import ResCNNDemapper2D
from pipeline.ofdm_channel_pipeline import OFDMChannelPipeline
from pipeline.self_supervised_trainer import SelfSupervisedTrainer
from benchmarks.classical_baseline import ClassicalBaseline
from benchmarks.complexity_profiler import ComplexityProfiler
from benchmarks.self_supervised_delta_bench import SelfSupervisedDeltaBenchmark
from benchmarks.pilot_ablation_bench import PilotAblationBenchmark
from plot_review_results import (
    plot_multipath_channel_comparison,
    plot_self_supervised_delta_comparison,
    plot_pilot_ablation_comparison,
    plot_computational_complexity_scaling
)

def run_master_experiments(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 88)
    print(" [MASTER SUITE] PROFESSOR REVIEW 1 - PHYSICAL LAYER EVALUATION")
    print(f"  * Execution Device : {device}")
    print(f"  * Output Directory : {out_dir}")
    print(f"  * Modulation       : 16-QAM (4 bits/symbol), Rate 1/2 LDPC (k=1792, 3584, 7168)")
    print("=" * 88)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Multipath Channel Models (AWGN vs Rayleigh vs TDL Ray-Tracing)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 88)
    print(" [EXPERIMENT 1] Realistic Multipath Channel Scaling (AWGN vs Rayleigh vs TDL Ray-Tracing)")
    print("#" * 88)
    
    ebno_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    channels_to_test = ["awgn", "rayleigh", "tdl_raytracing"]
    channel_results = {
        "ebno_range": ebno_range,
        "channels": {}
    }
    
    for ch in channels_to_test:
        print(f"\n---> Evaluating Channel Model: {ch.upper()} (k=1792, Nsc=64)...")
        # Pilot-Free Pipeline for Neural Demapper
        pipe_zero = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=4,
            coderate=0.5,
            channel_type=ch,
            pilot_mode="pilot_free",
            device=device
        )
        
        # Pilot-Assisted Pipeline for Classical Baseline
        pipe_pilot = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=4,
            coderate=0.5,
            channel_type=ch,
            pilot_mode="pilot_assisted",
            device=device
        )
        classical = ClassicalBaseline(pipe_pilot)
        
        model = ResCNNDemapper2D(num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3).to(device)
        trainer = SelfSupervisedTrainer(model, pipe_zero, device=device)
        
        print(f"  Training 2D ResCNN on {ch.upper()}...")
        trainer.supervised_train(steps=args.train_steps, batch_size=args.batch_size, verbose=False)
        
        ch_data = {
            "neural_raw_ber": [],
            "neural_post_fec_ber": [],
            "neural_bler": [],
            "classical_post_fec_ber": [],
            "classical_bler": []
        }
        
        for ebno in ebno_range:
            n_raw, n_fec, n_bler = trainer.evaluate(batch_size=args.eval_batches, ebno_db=float(ebno))
            _, c_fec, c_bler = classical.evaluate(batch_size=args.eval_batches, ebno_db=float(ebno))
            ch_data["neural_raw_ber"].append(n_raw)
            ch_data["neural_post_fec_ber"].append(n_fec)
            ch_data["neural_bler"].append(n_bler)
            ch_data["classical_post_fec_ber"].append(c_fec)
            ch_data["classical_bler"].append(c_bler)
            
        channel_results["channels"][ch] = ch_data
        print(f"  [DONE] Completed evaluation on {ch.upper()}.")

    # Save Channel Results
    json_path_ch = os.path.join(out_dir, "channel_model_comparison.json")
    with open(json_path_ch, "w") as f:
        json.dump(channel_results, f, indent=4)
    print(f"  [SAVED] Channel results to {json_path_ch}")
    plot_multipath_channel_comparison(channel_results, os.path.join(out_dir, "fig1_multipath_channel_scaling.png"))

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Self-Supervised Delta-Loop vs Non-Self-Supervised (Frozen)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 88)
    print(" [EXPERIMENT 2] Impact of Self-Supervised Delta Loop vs Non-Self-Supervised / Supervised")
    print("#" * 88)
    delta_bench = SelfSupervisedDeltaBenchmark(fft_size=64, channel_type="tdl_raytracing", device=device)
    delta_results = delta_bench.run_ablation(
        warmup_steps=args.train_steps,
        adapt_stream_steps=args.adapt_steps,
        ebno_range=ebno_range,
        eval_batches=args.eval_batches
    )
    
    json_path_delta = os.path.join(out_dir, "self_supervised_delta_ablation.json")
    with open(json_path_delta, "w") as f:
        json.dump(delta_results, f, indent=4)
    print(f"  [SAVED] Delta-loop ablation results to {json_path_delta}")
    plot_self_supervised_delta_comparison(delta_results, os.path.join(out_dir, "fig2_self_supervised_delta_impact.png"))

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Isolated Pilot Impact (With Pilots vs Without Pilots)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 88)
    print(" [EXPERIMENT 3] Isolated Impact of Neural Demapper (With Pilots vs Without Pilots)")
    print("#" * 88)
    pilot_bench = PilotAblationBenchmark(fft_size=64, channel_type="rayleigh", device=device)
    pilot_results = pilot_bench.run_pilot_ablation(
        train_steps=args.train_steps,
        ebno_range=ebno_range,
        eval_batches=args.eval_batches
    )
    
    json_path_pilot = os.path.join(out_dir, "pilot_ablation_results.json")
    with open(json_path_pilot, "w") as f:
        json.dump(pilot_results, f, indent=4)
    print(f"  [SAVED] Pilot ablation results to {json_path_pilot}")
    plot_pilot_ablation_comparison(pilot_results, os.path.join(out_dir, "fig3_isolated_pilot_impact.png"))

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: Computational Complexity Scaling (k=1792, 3584, 7168 bits)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 88)
    print(" [EXPERIMENT 4] Computational Complexity & Latency Scaling across Information Bit Sizes")
    print("#" * 88)
    profiler = ComplexityProfiler(device=device)
    complexity_results = profiler.profile_rescnn(
        num_subcarriers_list=[64, 128, 256],
        num_ofdm_symbols=14,
        num_bits_per_symbol=4,
        num_runs=30
    )
    profiler.print_complexity_table(complexity_results)
    
    json_path_comp = os.path.join(out_dir, "computational_complexity_scaling.json")
    with open(json_path_comp, "w") as f:
        json.dump(complexity_results, f, indent=4)
    print(f"  [SAVED] Complexity results to {json_path_comp}")
    plot_computational_complexity_scaling(complexity_results, os.path.join(out_dir, "fig4_computational_complexity_scaling.png"))

    print("\n" + "=" * 88)
    print(" [SUCCESS] ALL PROFESSOR REVIEW 1 EXPERIMENTS COMPLETED!")
    print(f"  * Generated JSON Data & Figures stored in: {out_dir}")
    print("=" * 88 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professor Review 1 Master Experiment Runner")
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--adapt_steps", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--snr_min", type=int, default=8)
    parser.add_argument("--snr_max", type=int, default=20)
    parser.add_argument("--snr_step", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=os.path.join(current_dir, "results"))
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    run_master_experiments(args)
