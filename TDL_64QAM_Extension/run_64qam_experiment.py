"""
Main Experiment Runner for 3GPP TDL & 64-QAM Neural Demodulation & LDPC.
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from TDL_64QAM_Extension.models.neural_64qam_demapper import Neural64QAMDemapper
from TDL_64QAM_Extension.pipeline.ofdm_64qam_tdl_chain import OFDM64QAMTDLPipeline
from TDL_64QAM_Extension.pipeline.trainer_64qam import JointTrainer64QAM
from TDL_64QAM_Extension.benchmarks.classical_64qam_baseline import Classical64QAMPilotBaseline
from TDL_64QAM_Extension.plot_64qam_results import (
    plot_64qam_ber_bler, plot_64qam_spectral_efficiency, plot_ece_reliability_diagram
)

def run_64qam_experiment(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("=" * 78)
    print(f" [START] 64-QAM 3GPP TR 38.901 TDL Neural Demodulation & 5G NR LDPC")
    print(f"  * Device: {device}")
    print(f"  * Modulation: 64-QAM (6 bits/symbol)")
    print(f"  * Channel: 3GPP TDL-A (3.5 GHz carrier, 36 km/h vehicular mobility)")
    print(f"  * Coded Block Length (n): 4,608 bits | Info Bits (k): 2,304 bits (Rate 1/2)")
    print(f"  * Training Steps: {args.steps}")
    print("=" * 78)

    os.makedirs(args.out_dir, exist_ok=True)

    pipeline = OFDM64QAMTDLPipeline(
        num_ofdm_symbols=14,
        fft_size=64,
        num_bits_per_symbol=6,
        coderate=args.coderate,
        num_ldpc_iter=args.ldpc_iter,
        tdl_model=args.tdl_model,
        speed=args.speed,
        device=device
    )

    neural_demapper = Neural64QAMDemapper(
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.res_blocks
    ).to(device)

    trainer = JointTrainer64QAM(
        demapper=neural_demapper,
        pipeline=pipeline,
        lr=args.lr,
        device=device
    )

    classical_baseline = Classical64QAMPilotBaseline(
        num_ofdm_symbols=14,
        fft_size=64,
        num_bits_per_symbol=6,
        coderate=args.coderate,
        num_ldpc_iter=args.ldpc_iter,
        tdl_model=args.tdl_model,
        speed=args.speed,
        device=device
    )

    # 1. Training Phase
    print(f"\n[Phase 1] Training Neural Demapper over 3GPP TDL Channel ({args.steps} steps)...")
    start_time = time.time()
    trainer.train_model(steps=args.steps, batch_size=args.batch_size)
    print(f"[OK] Training completed in {time.time() - start_time:.2f} seconds.")

    # Save weights
    model_path = os.path.join(args.out_dir, "neural_64qam_demapper_weights.pt")
    torch.save(neural_demapper.state_dict(), model_path)

    # 2. Benchmarking across Eb/N0
    print("\n[Phase 2] Benchmarking 64-QAM Performance over 3GPP TDL Channel...")
    ebno_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    results = {
        "ebno_range": ebno_range,
        "classical_raw_ber": [],
        "classical_post_fec_ber": [],
        "classical_bler": [],
        "neural_raw_ber": [],
        "neural_post_fec_ber": [],
        "neural_bler": []
    }

    eval_batches = args.eval_batches
    print(f"{'Eb/N0 (dB)':^11} | {'Classical Post-FEC':^18} | {'Neural Post-FEC':^18} | {'Neural BLER':^13}")
    print("-" * 78)

    for ebno in ebno_range:
        c_raw, c_fec, c_bler = classical_baseline.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        results["classical_raw_ber"].append(c_raw)
        results["classical_post_fec_ber"].append(c_fec)
        results["classical_bler"].append(c_bler)

        n_raw, n_fec, n_bler = trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        results["neural_raw_ber"].append(n_raw)
        results["neural_post_fec_ber"].append(n_fec)
        results["neural_bler"].append(n_bler)

        print(f" {ebno:4d} dB     | {c_fec:18.5e} | {n_fec:18.5e} | {n_bler:13.4f}")

    json_path = os.path.join(args.out_dir, "experiment_results_64qam_tdl.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[OK] Numeric JSON results saved to: {json_path}")

    # 3. Compute Calibration & Generate Figures
    print("\n[Phase 3] Computing LLR Calibration (ECE) and Generating Figures...")
    calibration_data = trainer.compute_ece_calibration(batch_size=64, ebno_db=18.0)
    print(f"  * Expected Calibration Error (ECE): {calibration_data['ece']:.4f}")

    plot_64qam_ber_bler(ebno_range, results, save_path=os.path.join(args.out_dir, "ber_bler_64qam_tdl.png"))
    plot_64qam_spectral_efficiency(ebno_range, results, save_path=os.path.join(args.out_dir, "spectral_efficiency_64qam.png"))
    plot_ece_reliability_diagram(calibration_data, save_path=os.path.join(args.out_dir, "llr_calibration_reliability.png"))

    print("\n" + "=" * 78)
    print(" [COMPLETED] 64-QAM 3GPP TDL Experiment Successfully Finished!")
    print("=" * 78)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="64-QAM 3GPP TDL Physical Layer Extension")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--ldpc_iter", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--res_blocks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tdl_model", type=str, default="A")
    parser.add_argument("--speed", type=float, default=10.0) # 36 km/h
    parser.add_argument("--snr_min", type=int, default=6)
    parser.add_argument("--snr_max", type=int, default=22)
    parser.add_argument("--snr_step", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=os.path.join(current_dir, "results"))
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    run_64qam_experiment(args)
