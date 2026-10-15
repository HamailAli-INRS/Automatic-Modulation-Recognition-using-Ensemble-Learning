"""
Publication-Quality Plotting Utilities for Joint Demodulation & FEC Experiments.
Generates vector/high-DPI PNGs for research papers.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_ieee_style():
    """Configure Matplotlib for IEEE Transactions paper style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'grid.alpha': 0.6,
        'grid.linestyle': '--'
    })

def plot_ber_bler_curves(ebno_range, results, save_path="ber_bler_comparison.png"):
    """
    Plot Raw BER and Post-FEC BER / BLER curves.
    """
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Bit Error Rate (BER)
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['classical_raw_ber']], 'r--o', label='Classical Pilot LS (Uncoded Demap)')
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['classical_post_fec_ber']], 'r-s', label='Classical Pilot + 5G LDPC (Post-FEC)')
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['neural_raw_ber']], 'b--^', label='Pilot-Free Neural Demapper (Raw)')
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['neural_post_fec_ber']], 'b-d', label='Proposed Pilot-Free + 5G LDPC (Post-FEC)')
    
    ax1.set_xlabel(r'$E_b/N_0$ (dB)')
    ax1.set_ylabel('Bit Error Rate (BER)')
    ax1.set_title('Bit Error Rate (BER) Performance')
    ax1.grid(True, which='both')
    ax1.set_ylim([1e-5, 1.0])
    ax1.legend(loc='lower left', framealpha=0.9)
    
    # 2. Block Error Rate (BLER)
    ax2.semilogy(ebno_range, [max(b, 1e-4) for b in results['classical_bler']], 'r-s', label='Classical Pilot + 5G LDPC')
    ax2.semilogy(ebno_range, [max(b, 1e-4) for b in results['neural_bler']], 'b-d', label='Proposed Pilot-Free + 5G LDPC')
    
    ax2.set_xlabel(r'$E_b/N_0$ (dB)')
    ax2.set_ylabel('Block Error Rate (BLER)')
    ax2.set_title('5G NR Frame Reliability (BLER)')
    ax2.grid(True, which='both')
    ax2.set_ylim([1e-4, 1.0])
    ax2.legend(loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] BER/BLER comparison plot saved to: {save_path}")

def plot_spectral_efficiency(ebno_range, results, save_path="spectral_efficiency_gain.png"):
    """
    Plot Effective Throughput / Spectral Efficiency Gain achieved by eliminating pilots.
    """
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Effective Throughput = (1 - BLER) * Modulation_Bits * CodeRate * (Data_Symbols / Total_Symbols)
    classical_throughput = (1.0 - np.array(results['classical_bler'])) * 4.0 * 0.5 * (12.0 / 14.0) # bits/s/Hz
    neural_throughput = (1.0 - np.array(results['neural_bler'])) * 4.0 * 0.5 * 1.0 # 100% data allocation
    
    ax.plot(ebno_range, classical_throughput, 'r--s', label='Classical System (14.3% Pilot Overhead)')
    ax.plot(ebno_range, neural_throughput, 'g-^', label='Proposed Pilot-Free Neural Demapper (0% Overhead)')
    
    # Fill throughput gain area
    ax.fill_between(ebno_range, classical_throughput, neural_throughput, color='lightgreen', alpha=0.3, label='Spectral Efficiency Gain (+16.7%)')
    
    ax.set_xlabel(r'$E_b/N_0$ (dB)')
    ax.set_ylabel('Effective Spectral Efficiency (bits / sec / Hz)')
    ax.set_title('Spectral Efficiency & Throughput Advantage (Zero-Pilot Transmission)')
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Spectral Efficiency plot saved to: {save_path}")

def plot_learning_curves(loss_history, save_path="online_adaptation_loss.png"):
    """
    Plot the loss convergence of the self-supervised Delta feedback loop.
    """
    set_ieee_style()
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    
    iterations = range(1, len(loss_history['instant_loss']) + 1)
    
    ax1.plot(iterations, loss_history['instant_loss'], color='orange', alpha=0.4, label='Instantaneous Frame Loss')
    ax1.plot(iterations, loss_history['buffer_loss'], color='blue', linewidth=2.0, label='Buffer Replay Loss (Δ Loop)')
    
    ax1.set_xlabel('Online Adaptation Steps')
    ax1.set_ylabel('Self-Supervised BCE Loss (Δ)')
    ax1.set_title('Online Decision-Directed Continuous Adaptation')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Online Adaptation plot saved to: {save_path}")
