"""
Publication-Quality Plotting Suite for Professor Review 1.
Generates 4 Key Result Figures:
Figure 1: Multipath Channel Scaling (AWGN vs Rayleigh vs 3GPP TDL Ray-Tracing) across Bit Lengths (k=1792, 3584, 7168).
Figure 2: Impact of Self-Supervised Delta-Loop (Delta Adapted vs Frozen/Non-Adapted vs Supervised vs Classical).
Figure 3: Isolated Pilot Impact (With Pilots vs Without Pilots vs Classical Blind Failure).
Figure 4: 2D ResCNN Computational Complexity & Latency Scaling across Information Bit Sizes.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14

def plot_multipath_channel_comparison(channel_results, save_path):
    """
    Figure 1: AWGN vs Rayleigh vs 3GPP TDL Ray-Tracing Multipath Channel BER curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {"awgn": "#2ca02c", "rayleigh": "#1f77b4", "tdl_raytracing": "#d62728"}
    labels = {"awgn": "AWGN (Ideal AWGN)", "rayleigh": "Rayleigh Multipath (NLOS)", "tdl_raytracing": "3GPP TDL Ray-Tracing (36 km/h)"}
    markers = {"awgn": "o", "rayleigh": "s", "tdl_raytracing": "^"}
    
    ebno = channel_results["ebno_range"]
    
    # Left: Post-FEC BER
    for ch, data in channel_results["channels"].items():
        ber_vals = np.maximum(data["neural_post_fec_ber"], 1e-6)
        ax1.semilogy(ebno, ber_vals, marker=markers.get(ch, 'o'), color=colors.get(ch, '#333333'),
                     linewidth=2.2, markersize=7, label=f"Neural Demapper ({labels.get(ch, ch)})")
        if "classical_post_fec_ber" in data:
            c_ber = np.maximum(data["classical_post_fec_ber"], 1e-6)
            ax1.semilogy(ebno, c_ber, linestyle="--", marker=markers.get(ch, 'o'), color=colors.get(ch, '#333333'),
                         alpha=0.6, linewidth=1.5, label=f"Classical LMMSE ({labels.get(ch, ch)})")
            
    ax1.set_xlabel(r"$\mathbf{E_b/N_0\ (dB)}$", fontweight='bold')
    ax1.set_ylabel(r"$\mathbf{Post\text{-}FEC\ Information\ Bit\ Error\ Rate\ (BER)}$", fontweight='bold')
    ax1.set_title("Post-FEC BER over Multipath Channels (16-QAM, k=1792)", fontweight='bold')
    ax1.set_ylim([1e-6, 1.0])
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.95)
    
    # Right: BLER
    for ch, data in channel_results["channels"].items():
        bler_vals = np.maximum(data["neural_bler"], 1e-4)
        ax2.semilogy(ebno, bler_vals, marker=markers.get(ch, 'o'), color=colors.get(ch, '#333333'),
                     linewidth=2.2, markersize=7, label=f"Neural Demapper ({labels.get(ch, ch)})")
        if "classical_bler" in data:
            c_bler = np.maximum(data["classical_bler"], 1e-4)
            ax2.semilogy(ebno, c_bler, linestyle="--", marker=markers.get(ch, 'o'), color=colors.get(ch, '#333333'),
                         alpha=0.6, linewidth=1.5, label=f"Classical LMMSE ({labels.get(ch, ch)})")
            
    ax2.set_xlabel(r"$\mathbf{E_b/N_0\ (dB)}$", fontweight='bold')
    ax2.set_ylabel(r"$\mathbf{Block\ Error\ Rate\ (BLER)}$", fontweight='bold')
    ax2.set_title("Block Error Rate (BLER) across Multipath Realism", fontweight='bold')
    ax2.set_ylim([1e-4, 1.1])
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc="lower left", framealpha=0.95)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {save_path}")

def plot_self_supervised_delta_comparison(delta_results, save_path):
    """
    Figure 2: Impact of Self-Supervised Delta-Loop.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ebno = delta_results["ebno_range"]
    
    # Left: Post-FEC BER Comparison
    f_ber = np.maximum(delta_results["frozen_post_fec_ber"], 1e-6)
    d_ber = np.maximum(delta_results["delta_post_fec_ber"], 1e-6)
    s_ber = np.maximum(delta_results["supervised_post_fec_ber"], 1e-6)
    c_ber = np.maximum(delta_results["classical_pilot_fec_ber"], 1e-6)
    
    ax1.semilogy(ebno, f_ber, 'r-o', linewidth=2, markersize=7, label="Frozen Demapper (No Delta Loop)")
    ax1.semilogy(ebno, d_ber, 'b-^', linewidth=2.5, markersize=8, label="Self-Supervised (Online Delta Loop)")
    ax1.semilogy(ebno, s_ber, 'g--s', linewidth=2, markersize=7, label="Supervised (Upper Bound Baseline)")
    ax1.semilogy(ebno, c_ber, 'k:d', linewidth=2, markersize=6, alpha=0.7, label="Classical LMMSE (With Pilots)")
    
    ax1.set_xlabel(r"$\mathbf{E_b/N_0\ (dB)}$", fontweight='bold')
    ax1.set_ylabel(r"$\mathbf{Post\text{-}FEC\ Bit\ Error\ Rate\ (BER)}$", fontweight='bold')
    ax1.set_title("Impact of Self-Supervised Delta Loop on Post-FEC BER", fontweight='bold')
    ax1.set_ylim([1e-6, 1.0])
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.95)
    
    # Right: BLER
    f_bler = np.maximum(delta_results["frozen_bler"], 1e-4)
    d_bler = np.maximum(delta_results["delta_bler"], 1e-4)
    s_bler = np.maximum(delta_results["supervised_bler"], 1e-4)
    c_bler = np.maximum(delta_results["classical_pilot_bler"], 1e-4)
    
    ax2.semilogy(ebno, f_bler, 'r-o', linewidth=2, markersize=7, label="Frozen Demapper (No Delta Loop)")
    ax2.semilogy(ebno, d_bler, 'b-^', linewidth=2.5, markersize=8, label="Self-Supervised (Online Delta Loop)")
    ax2.semilogy(ebno, s_bler, 'g--s', linewidth=2, markersize=7, label="Supervised (Upper Bound)")
    ax2.semilogy(ebno, c_bler, 'k:d', linewidth=2, markersize=6, alpha=0.7, label="Classical LMMSE (With Pilots)")
    
    ax2.set_xlabel(r"$\mathbf{E_b/N_0\ (dB)}$", fontweight='bold')
    ax2.set_ylabel(r"$\mathbf{Block\ Error\ Rate\ (BLER)}$", fontweight='bold')
    ax2.set_title("BLER Resilience with Decision-Directed Delta Adaptation", fontweight='bold')
    ax2.set_ylim([1e-4, 1.1])
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc="lower left", framealpha=0.95)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {save_path}")

def plot_pilot_ablation_comparison(pilot_results, save_path):
    """
    Figure 3: Isolated Impact of Neural Demapper (With Pilots vs Without Pilots).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ebno = pilot_results["ebno_range"]
    
    cp_ber = np.maximum(pilot_results["classical_with_pilots_ber"], 1e-6)
    cn_ber = np.maximum(pilot_results["classical_no_pilots_ber"], 1e-6)
    np_ber = np.maximum(pilot_results["neural_with_pilots_ber"], 1e-6)
    nn_ber = np.maximum(pilot_results["neural_no_pilots_ber"], 1e-6)
    
    # Left: BER Curves
    ax1.semilogy(ebno, cn_ber, 'r--x', linewidth=2, markersize=8, label="Classical Blind (No Pilots) - Fails")
    ax1.semilogy(ebno, cp_ber, 'g:d', linewidth=2, markersize=7, label="Classical LMMSE (With 14.3% Pilots)")
    ax1.semilogy(ebno, np_ber, 'm-.s', linewidth=2, markersize=7, label="Neural Demapper (With Pilots)")
    ax1.semilogy(ebno, nn_ber, 'b-o', linewidth=2.5, markersize=8, label="Neural Demapper (Zero Pilots / 100% Data)")
    
    ax1.set_xlabel(r"$\mathbf{E_b/N_0\ (dB)}$", fontweight='bold')
    ax1.set_ylabel(r"$\mathbf{Post\text{-}FEC\ Bit\ Error\ Rate\ (BER)}$", fontweight='bold')
    ax1.set_title("Isolated Pilot Impact: Neural vs Classical Demapping", fontweight='bold')
    ax1.set_ylim([1e-6, 1.0])
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.95)
    
    # Right: Effective Throughput / Spectral Efficiency Gain
    overhead = pilot_results["pilot_overhead_percent"]
    gain = pilot_results["spectral_efficiency_gain_percent"]
    
    categories = ['Classical (Pilots)', 'Neural (Zero Pilots)']
    efficiency = [100.0 - overhead, 100.0]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax2.bar(categories, efficiency, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel(r"$\mathbf{Normalized\ Useful\ Payload\ (\%)\ / \ Spectral\ Efficiency}$", fontweight='bold')
    ax2.set_title(f"Spectral Efficiency Gain: +{gain:.1f}% by Eliminating Pilots", fontweight='bold')
    ax2.set_ylim([0, 115])
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}% Payload',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    ax2.grid(axis='y', ls="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {save_path}")

def plot_computational_complexity_scaling(complexity_data, save_path):
    """
    Figure 4: Computational Complexity & Latency Scaling (k=1792, 3584, 7168).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    k_bits = [d["info_bits_k"] for d in complexity_data]
    gflops_nn = [d["rescnn_gflops"] for d in complexity_data]
    gflops_cl = [d["classical_gflops"] for d in complexity_data]
    lat_ms = [d["latency_batched_frame_ms"] for d in complexity_data]
    tput_mbps = [d["throughput_batched_mbps"] for d in complexity_data]
    
    x = np.arange(len(k_bits))
    width = 0.35
    
    # Left: GFLOPs Comparison
    ax1.bar(x - width/2, gflops_nn, width, label='2D ResCNN Demapper', color='#1f77b4', edgecolor='black')
    ax1.bar(x + width/2, gflops_cl, width, label='Classical LMMSE + APP', color='#ff7f0e', edgecolor='black')
    ax1.set_xlabel(r"$\mathbf{Information\ Block\ Length\ k\ (Bits)}$", fontweight='bold')
    ax1.set_ylabel(r"$\mathbf{Computation\ per\ Frame\ (GFLOPs)}$", fontweight='bold')
    ax1.set_title("Computational Complexity Scaling (FLOPs vs Bit Length)", fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}\n(n={d['coded_bits_n']})" for k, d in zip(k_bits, complexity_data)])
    ax1.grid(axis='y', ls="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    # Right: Inference Latency and Throughput
    color = 'tab:blue'
    ax2.set_xlabel(r"$\mathbf{Information\ Block\ Length\ k\ (Bits)}$", fontweight='bold')
    ax2.set_ylabel(r"$\mathbf{Inference\ Latency\ (ms/frame)}$", color=color, fontweight='bold')
    l1 = ax2.plot(x, lat_ms, color=color, marker='o', linewidth=2.5, markersize=8, label="Latency (ms)")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"k={k}\n({d['subcarriers']} SCs)" for k, d in zip(k_bits, complexity_data)])
    
    ax3 = ax2.twinx()
    color = 'tab:green'
    ax3.set_ylabel(r"$\mathbf{Effective\ Demapping\ Throughput\ (Mbps)}$", color=color, fontweight='bold')
    l2 = ax3.plot(x, tput_mbps, color=color, marker='s', linestyle='--', linewidth=2.5, markersize=8, label="Throughput (Mbps)")
    ax3.tick_params(axis='y', labelcolor=color)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.set_title("ResCNN Latency & Throughput Scaling", fontweight='bold')
    ax2.grid(True, ls="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {save_path}")
