"""
Main Experiment Runner for Joint Neural Demodulation and Advanced FEC (5G NR LDPC) over OFDM.
Executes the Pilot-Free Self-Supervised online training loop with Experience Replay Buffer (Delta Feedback)
and generates benchmarking results against classical pilot-assisted transmission.
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np

# Ensure project root is in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Joint_Neural_Demod_FEC.models.neural_demapper import ResCNNDemapper
from Joint_Neural_Demod_FEC.models.replay_buffer import ExperienceReplayBuffer
from Joint_Neural_Demod_FEC.pipeline.ofdm_fec_chain import OFDMFECPipeline
from Joint_Neural_Demod_FEC.pipeline.trainer import JointDemodFECTrainer
from Joint_Neural_Demod_FEC.benchmarks.classical_baseline import ClassicalPilotBaseline
from Joint_Neural_Demod_FEC.plot_results import plot_ber_bler_curves, plot_spectral_efficiency, plot_learning_curves

def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("=" * 78)
    print(f" [START] Pilot-Free Joint Neural Demodulation & 5G NR LDPC Simulation")
    print(f"  * Device: {device}")
    print(f"  * Modulation: 16-QAM ({args.bits_per_symbol} bits/symbol)")
    print(f"  * Channel: {args.channel.upper()} (Multipath Fading / Noise)")
    print(f"  * Warmup Steps: {args.warmup_steps}")
    print(f"  * Online Self-Supervised Steps: {args.steps}")
    print(f"  * Buffer Capacity: {args.buffer_capacity} frames")
    print(f"  * 5G NR LDPC Code Rate: {args.coderate}")
    print("=" * 78)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Initialize Pilot-Free Transmission Chain & Neural Demapper
    pipeline = OFDMFECPipeline(
        num_ofdm_symbols=14,
        fft_size=args.fft_size,
        num_bits_per_symbol=args.bits_per_symbol,
        coderate=args.coderate,
        num_ldpc_iter=args.ldpc_iter,
        channel_type=args.channel,
        device=device
    )

    print(f"  * Subcarriers: {args.fft_size} | Coded Block Length (n): {pipeline.n:,} bits | Transmitted Info Bits (k): {pipeline.k:,} bits")

    neural_demapper = ResCNNDemapper(
        num_bits_per_symbol=args.bits_per_symbol,
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.res_blocks
    ).to(device)

    replay_buffer = ExperienceReplayBuffer(capacity=args.buffer_capacity)

    trainer = JointDemodFECTrainer(
        demapper=neural_demapper,
        pipeline=pipeline,
        replay_buffer=replay_buffer,
        lr=args.lr,
        buffer_batch_size=args.batch_size,
        device=device
    )

    # 2. Initialize Classical Pilot Baseline for Direct Benchmarking
    classical_baseline = ClassicalPilotBaseline(
        num_ofdm_symbols=14,
        fft_size=args.fft_size,
        num_bits_per_symbol=args.bits_per_symbol,
        coderate=args.coderate,
        num_ldpc_iter=args.ldpc_iter,
        channel_type=args.channel,
        device=device
    )

    # 3. Phase 1: Bootstrap Warmup
    if args.warmup_steps > 0:
        print(f"\n[Phase 1] Geometric Initialization Warmup ({args.warmup_steps} steps)...")
        trainer.bootstrap_warmup(steps=args.warmup_steps, batch_size=args.batch_size)
        print("[OK] Warmup completed.")

    # 4. Phase 2: Online Self-Supervised Adaptation Loop (The Whiteboard Delta Feedback Loop)
    print("\n[Phase 2] Running Pilot-Free Self-Supervised Online Adaptation (Delta Loop)...")
    loss_history = {"instant_loss": [], "buffer_loss": []}
    start_time = time.time()

    for step in range(1, args.steps + 1):
        # Transmit pilot-free bursts over varying channel SNR conditions (6 to 18 dB)
        rand_ebno = float(np.random.uniform(8.0, 18.0))
        _, _, y_grid, _ = pipeline.transmit(args.batch_size, rand_ebno)
        
        metrics = trainer.train_step(y_grid, rand_ebno)
        loss_history["instant_loss"].append(metrics["instant_loss"])
        loss_history["buffer_loss"].append(metrics["buffer_loss"])

        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f" Step {step:4d}/{args.steps} | Buffer: {metrics['buffer_size']:3d} | "
                  f"Inst Loss: {metrics['instant_loss']:.4f} | Buffer Loss (Delta): {metrics['buffer_loss']:.4f} | "
                  f"Raw Disagree: {metrics['raw_disagreement']*100:.1f}%")

    train_duration = time.time() - start_time
    print(f"[OK] Online adaptation completed in {train_duration:.2f} seconds.")

    # Save model weights safely
    model_path = os.path.join(args.out_dir, "neural_demapper_weights.pt")
    try:
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception:
                pass
        torch.save(neural_demapper.state_dict(), model_path)
        print(f"[OK] Model weights saved to: {model_path}")
    except Exception as e:
        print(f"[WARN] Could not overwrite model weights file ({e}), continuing...")

    # 5. Phase 3: Comprehensive Benchmarking vs Classical Pilot Baseline
    print("\n[Phase 3] Evaluating BER / BLER Benchmarks across Eb/N0 Range...")
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
    print(f"{'Eb/N0 (dB)':^11} | {'Classical Post-FEC':^18} | {'Proposed Pilot-Free Post-FEC':^28} | {'Proposed BLER':^13}")
    print("-" * 78)

    for ebno in ebno_range:
        # Evaluate Classical Pilot Baseline
        c_raw, c_fec, c_bler = classical_baseline.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        results["classical_raw_ber"].append(c_raw)
        results["classical_post_fec_ber"].append(c_fec)
        results["classical_bler"].append(c_bler)

        # Evaluate Proposed Pilot-Free Joint Neural Demapper
        n_raw, n_fec, n_bler = trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        results["neural_raw_ber"].append(n_raw)
        results["neural_post_fec_ber"].append(n_fec)
        results["neural_bler"].append(n_bler)

        print(f" {ebno:4d} dB     | {c_fec:18.5e} | {n_fec:28.5e} | {n_bler:13.4f}")

    # Save numeric JSON results
    json_path = os.path.join(args.out_dir, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[OK] Results JSON exported to: {json_path}")

    # 6. Phase 4: Generate Publication-Ready Figures
    print("\n[Phase 4] Generating Publication-Ready Figures...")
    plot_ber_bler_curves(ebno_range, results, save_path=os.path.join(args.out_dir, "ber_bler_comparison.png"))
    plot_spectral_efficiency(ebno_range, results, save_path=os.path.join(args.out_dir, "spectral_efficiency_gain.png"))
    plot_learning_curves(loss_history, save_path=os.path.join(args.out_dir, "online_adaptation_loss.png"))

    print("\n" + "=" * 78)
    print(" [COMPLETED] All experiments finished and figures generated successfully!")
    print("=" * 78)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot-Free Joint Neural Demodulation & 5G NR LDPC Experiments")
    parser.add_argument("--fft_size", type=int, default=64, help="Number of OFDM subcarriers (64 for k=1792, 128 for k=3584, 256 for k=7168)")
    parser.add_argument("--warmup_steps", type=int, default=80, help="Number of initialization warmup steps")
    parser.add_argument("--steps", type=int, default=150, help="Number of online adaptation steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--eval_batches", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--buffer_capacity", type=int, default=128, help="Experience replay buffer capacity")
    parser.add_argument("--bits_per_symbol", type=int, default=4, help="Bits per symbol (4 for 16-QAM)")
    parser.add_argument("--coderate", type=float, default=0.5, help="LDPC Code Rate")
    parser.add_argument("--ldpc_iter", type=int, default=20, help="LDPC BP iterations")
    parser.add_argument("--hidden_dim", type=int, default=48, help="Hidden channels in ResCNNDemapper")
    parser.add_argument("--res_blocks", type=int, default=3, help="Number of residual blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--channel", type=str, default="awgn", choices=["rayleigh", "awgn"], help="Channel type")
    parser.add_argument("--snr_min", type=int, default=0, help="Minimum Eb/N0 in dB")
    parser.add_argument("--snr_max", type=int, default=16, help="Maximum Eb/N0 in dB")
    parser.add_argument("--snr_step", type=int, default=2, help="Eb/N0 step in dB")
    parser.add_argument("--out_dir", type=str, default=os.path.join(current_dir, "results"), help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    run_experiment(args)
