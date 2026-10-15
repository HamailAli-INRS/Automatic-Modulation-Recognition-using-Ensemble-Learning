"""
Plotting utilities for 64-QAM 3GPP TDL experiments.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_ieee_style():
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

def plot_64qam_ber_bler(ebno_range, results, save_path="ber_bler_64qam_tdl.png"):
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. BER
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['classical_raw_ber']], 'r--o', label='Classical LS (Uncoded Demap)')
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['classical_post_fec_ber']], 'r-s', label='Classical LS + 5G LDPC (Post-FEC)')
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['neural_raw_ber']], 'b--^', label='Neural Demapper (Raw Demap)')
    ax1.semilogy(ebno_range, [max(b, 1e-5) for b in results['neural_post_fec_ber']], 'b-d', label='Proposed Neural + 5G LDPC (Post-FEC)')
    
    ax1.set_xlabel(r'$E_b/N_0$ (dB)')
    ax1.set_ylabel('Bit Error Rate (BER)')
    ax1.set_title('64-QAM BER over 3GPP TR 38.901 TDL-A Channel')
    ax1.grid(True, which='both')
    ax1.set_ylim([1e-5, 1.0])
    ax1.legend(loc='lower left', framealpha=0.9)
    
    # 2. BLER
    ax2.semilogy(ebno_range, [max(b, 1e-4) for b in results['classical_bler']], 'r-s', label='Classical LS + 5G LDPC')
    ax2.semilogy(ebno_range, [max(b, 1e-4) for b in results['neural_bler']], 'b-d', label='Proposed Neural + 5G LDPC')
    
    ax2.set_xlabel(r'$E_b/N_0$ (dB)')
    ax2.set_ylabel('Block Error Rate (BLER)')
    ax2.set_title('64-QAM 5G NR Frame Reliability (BLER)')
    ax2.grid(True, which='both')
    ax2.set_ylim([1e-4, 1.0])
    ax2.legend(loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] 64-QAM BER/BLER plot saved to: {save_path}")

def plot_64qam_spectral_efficiency(ebno_range, results, save_path="spectral_efficiency_64qam.png"):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Effective Throughput = (1 - BLER) * 6 bits/sym * 0.5 * (768 / 896)
    classical_throughput = (1.0 - np.array(results['classical_bler'])) * 6.0 * 0.5 * (768.0 / 896.0)
    neural_throughput = (1.0 - np.array(results['neural_bler'])) * 6.0 * 0.5 * (768.0 / 896.0)
    
    ax.plot(ebno_range, classical_throughput, 'r--s', label='Classical 64-QAM (LS Estimation)')
    ax.plot(ebno_range, neural_throughput, 'b-d', label='Proposed Neural 64-QAM Demapper')
    
    ax.set_xlabel(r'$E_b/N_0$ (dB)')
    ax.set_ylabel('Effective Spectral Efficiency (bits / sec / Hz)')
    ax.set_title('64-QAM Throughput Performance (3GPP TDL Channel)')
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] 64-QAM Spectral Efficiency plot saved to: {save_path}")

def plot_ece_reliability_diagram(calibration_data, save_path="llr_calibration_reliability.png"):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    
    confs = calibration_data['bin_confidences']
    accs = calibration_data['bin_accuracies']
    ece = calibration_data['ece']
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(confs, accs, 'b-o', label=f'Neural Demapper (ECE = {ece:.4f})')
    ax.bar(confs, [abs(a - c) for a, c in zip(accs, confs)], width=0.06, alpha=0.2, color='red', label='Calibration Gap')
    
    ax.set_xlabel('Mean Confidence $P(b=1)$')
    ax.set_ylabel('Empirical Accuracy')
    ax.set_title('LLR Calibration Reliability Diagram (Objective 3)')
    ax.grid(True)
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] LLR Calibration plot saved to: {save_path}")
