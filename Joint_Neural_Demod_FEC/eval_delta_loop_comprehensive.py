"""
Comprehensive Self-Supervised Delta-Loop Multi-Condition Benchmark Script.
Evaluates Delta Loop ON vs Delta Loop OFF (Frozen) vs Supervised Upper Bound vs Classical
across:
1. Baseline AWGN (k=1792 bits)
2. Scaled Info Bits in AWGN (k=1792, 3584, 7168, 8400 bits)
3. Rayleigh Multipath Fading (k=1792, 3584, 7168, 8400 bits)
"""

import os
import sys
import copy
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

project_root = r"c:\Users\hamhamai001\Desktop\Amr using EL"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Joint_Neural_Demod_FEC.models.neural_demapper import ResCNNDemapper
from Joint_Neural_Demod_FEC.models.replay_buffer import ExperienceReplayBuffer
from Joint_Neural_Demod_FEC.pipeline.ofdm_fec_chain import OFDMFECPipeline
from Joint_Neural_Demod_FEC.pipeline.trainer import JointDemodFECTrainer
from Joint_Neural_Demod_FEC.benchmarks.classical_baseline import ClassicalPilotBaseline

def run_delta_loop_evaluation(channel_type="awgn", fft_size=64, warmup_steps=50, delta_steps=60, eval_batches=16):
    device = "cpu"
    ebno_range = [6, 8, 10, 12, 14, 16, 18]
    
    pipeline = OFDMFECPipeline(
        num_ofdm_symbols=14,
        fft_size=fft_size,
        num_bits_per_symbol=4,
        coderate=0.5,
        num_ldpc_iter=20,
        channel_type=channel_type,
        device=device
    )
    
    classical_baseline = ClassicalPilotBaseline(
        num_ofdm_symbols=14,
        fft_size=fft_size,
        num_bits_per_symbol=4,
        coderate=0.5,
        num_ldpc_iter=20,
        channel_type=channel_type,
        device=device
    )
    
    # 1. Base Model for Warmup
    base_demapper = ResCNNDemapper(num_bits_per_symbol=4, hidden_dim=48, num_res_blocks=3).to(device)
    replay_buffer = ExperienceReplayBuffer(capacity=128)
    trainer = JointDemodFECTrainer(base_demapper, pipeline, replay_buffer, lr=1e-3, device=device)
    
    # Initial Warmup
    if warmup_steps > 0:
        trainer.bootstrap_warmup(steps=warmup_steps, batch_size=16)
        
    # 2. Delta Loop OFF (Frozen model)
    frozen_demapper = copy.deepcopy(base_demapper)
    frozen_trainer = JointDemodFECTrainer(frozen_demapper, pipeline, ExperienceReplayBuffer(16), lr=1e-3, device=device)
    
    # 3. Supervised Upper Bound (Genie-Aided)
    supervised_demapper = copy.deepcopy(base_demapper)
    supervised_opt = torch.optim.Adam(supervised_demapper.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(delta_steps):
        u, c, y, _ = pipeline.transmit(16, ebno_db=12.0)
        supervised_opt.zero_grad()
        llrs = supervised_demapper(y)
        loss = bce(llrs, c)
        loss.backward()
        supervised_opt.step()
    supervised_trainer = JointDemodFECTrainer(supervised_demapper, pipeline, ExperienceReplayBuffer(16), lr=1e-3, device=device)
    
    # 4. Delta Loop ON (Self-Supervised Adaptation)
    delta_demapper = copy.deepcopy(base_demapper)
    delta_buffer = ExperienceReplayBuffer(capacity=128)
    delta_trainer = JointDemodFECTrainer(delta_demapper, pipeline, delta_buffer, lr=1e-3, device=device)
    adaptation_metrics = {"inst_loss": [], "buffer_loss": [], "disagree": []}
    
    for s in range(delta_steps):
        rand_ebno = float(np.random.uniform(8.0, 16.0))
        _, _, y, _ = pipeline.transmit(16, rand_ebno)
        m = delta_trainer.train_step(y, rand_ebno)
        adaptation_metrics["inst_loss"].append(m["instant_loss"])
        adaptation_metrics["buffer_loss"].append(m["buffer_loss"])
        adaptation_metrics["disagree"].append(m["raw_disagreement"])
        
    # 5. Evaluate all 4 across SNR
    res = {
        "channel": channel_type,
        "fft_size": fft_size,
        "k": pipeline.k,
        "n": pipeline.n,
        "ebno_range": ebno_range,
        "frozen_post_fec": [],
        "delta_post_fec": [],
        "supervised_post_fec": [],
        "classical_post_fec": [],
        "adaptation_metrics": adaptation_metrics
    }
    
    for ebno in ebno_range:
        _, c_fec, _ = classical_baseline.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        _, f_fec, _ = frozen_trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        _, d_fec, _ = delta_trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        _, s_fec, _ = supervised_trainer.evaluate(batch_size=eval_batches, ebno_db=float(ebno))
        
        res["classical_post_fec"].append(c_fec)
        res["frozen_post_fec"].append(f_fec)
        res["delta_post_fec"].append(d_fec)
        res["supervised_post_fec"].append(s_fec)
        
    return res

def main():
    out_dir = os.path.join(project_root, "Joint_Neural_Demod_FEC", "results", "delta_loop_study")
    os.makedirs(out_dir, exist_ok=True)
    
    all_study_results = {}
    
    # Study 1: Baseline AWGN (k=1792)
    print("=== Running Study 1: Baseline AWGN (k=1792, fft=64) ===")
    all_study_results["awgn_k1792"] = run_delta_loop_evaluation(channel_type="awgn", fft_size=64)
    
    # Study 2: Bit Scaling in AWGN (k=3584, 7168, 8400)
    for fft, k_val in [(128, 3584), (256, 7168), (300, 8400)]:
        print(f"=== Running Study 2: Scaled Bits AWGN (k={k_val}, fft={fft}) ===")
        all_study_results[f"awgn_k{k_val}"] = run_delta_loop_evaluation(channel_type="awgn", fft_size=fft, warmup_steps=40, delta_steps=40)
        
    # Study 3: Rayleigh Channel Benchmarking (k=1792, 3584, 7168, 8400)
    for fft, k_val in [(64, 1792), (128, 3584), (256, 7168), (300, 8400)]:
        print(f"=== Running Study 3: Rayleigh Fading (k={k_val}, fft={fft}) ===")
        all_study_results[f"rayleigh_k{k_val}"] = run_delta_loop_evaluation(channel_type="rayleigh", fft_size=fft, warmup_steps=40, delta_steps=40)
        
    # Save combined results JSON
    json_path = os.path.join(out_dir, "delta_loop_comprehensive_study.json")
    with open(json_path, "w") as f:
        json.dump(all_study_results, f, indent=4)
    print(f"[OK] Full Delta-Loop Study JSON saved to: {json_path}")
    
    # Generate Plots
    generate_study_plots(all_study_results, out_dir)

def generate_study_plots(data, out_dir):
    ebno = data["awgn_k1792"]["ebno_range"]
    
    # Figure 1: AWGN Delta ON vs OFF vs Supervised vs Classical (k=1792)
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(ebno, data["awgn_k1792"]["frozen_post_fec"], 'r--o', label="Delta Loop OFF (Frozen)", linewidth=2)
    plt.plot(ebno, data["awgn_k1792"]["classical_post_fec"], 'g-^', label="Classical Pilot Baseline", linewidth=2)
    plt.plot(ebno, data["awgn_k1792"]["delta_post_fec"], 'b-s', label="Delta Loop ON (Self-Supervised)", linewidth=2.5)
    plt.plot(ebno, data["awgn_k1792"]["supervised_post_fec"], 'k:*', label="Supervised (Upper Bound)", linewidth=1.5)
    plt.yscale('log')
    plt.ylim([1e-5, 1.0])
    plt.xlabel("Eb/N0 (dB)", fontsize=11, fontweight='bold')
    plt.ylabel("Post-FEC BER", fontsize=11, fontweight='bold')
    plt.title("Self-Supervised Delta-Loop Ablation: AWGN (k=1,792 bits)", fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_delta_ablation_awgn_k1792.png"))
    plt.close()
    
    # Figure 2: Convergence Dynamics (Disagreement & Loss)
    m = data["awgn_k1792"]["adaptation_metrics"]
    steps = list(range(1, len(m["inst_loss"]) + 1))
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    ax1.plot(steps, m["inst_loss"], 'b-', label="Instantaneous Loss", linewidth=1.5)
    ax1.plot(steps, m["buffer_loss"], 'm--', label="Replay Buffer Loss (Delta)", linewidth=2)
    ax1.set_xlabel("Online Adaptation Steps", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Cross-Entropy Loss", color='b', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2 = ax1.twinx()
    disagree_pct = [d * 100 for d in m["disagree"]]
    ax2.plot(steps, disagree_pct, 'r-.', label="Demapper-FEC Disagreement (%)", linewidth=2)
    ax2.set_ylabel("Raw Disagreement Rate (%)", color='r', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title("Online Delta-Loop Convergence & Disagreement Elimination", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_delta_convergence_dynamics.png"))
    plt.close()
    
    # Figure 3: Bit Scaling Delta Loop Gain (k=1792 to 8400)
    plt.figure(figsize=(7, 5), dpi=300)
    k_keys = ["awgn_k1792", "awgn_k3584", "awgn_k7168", "awgn_k8400"]
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ["k=1,792 bits", "k=3,584 bits", "k=7,168 bits", "k=8,400 bits"]
    
    for key, c, lab in zip(k_keys, colors_list, labels):
        plt.plot(ebno, data[key]["delta_post_fec"], color=c, marker='s', linewidth=2, label=f"Delta ON ({lab})")
        plt.plot(ebno, data[key]["frozen_post_fec"], color=c, linestyle='--', marker='o', alpha=0.5, label=f"Delta OFF ({lab})")
        
    plt.yscale('log')
    plt.ylim([1e-5, 1.0])
    plt.xlabel("Eb/N0 (dB)", fontsize=11, fontweight='bold')
    plt.ylabel("Post-FEC BER", fontsize=11, fontweight='bold')
    plt.title("Delta Loop Impact Across Information Bit Block Lengths (AWGN)", fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_delta_bit_scaling_comparison.png"))
    plt.close()

if __name__ == "__main__":
    main()
