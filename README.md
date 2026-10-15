# 5G NR Joint Neural Demodulation & LDPC Decoding via Decision-Directed Self-Supervised Learning

An AI-native physical layer receiver for **5G New Radio (NR) and 6G OFDM systems** that eliminates pilot overhead via a **2D Residual CNN Demapper (2D ResCNN)** trained through **decision-directed self-supervised learning** with 5G NR LDPC codes.

---

## 🎯 Executive Highlights & Key Findings

| Metric / Parameter | Value / Achievement | Practical Significance |
|---|:---:|---|
| **Spectral Efficiency Gain** | **+16.67%** (2.000 vs. 1.714 bits/s/Hz) | 100% data payload, 0% pilot overhead across all block lengths. |
| **Low-SNR Waterfall Gain** | **+2.0 dB to +3.8 dB** vs. 3GPP Baseline | Zero post-FEC BER at 6 dB where classical baseline BLER is 65%. |
| **Self-Supervised Adaptation** | Disagreement dropped **26.8% -> 0.1%** | Closes within 0.2 dB of genie-aided bound with **zero true labels**. |
| **Maximum Block Scaling** | **k = 8,400 bits** (N_sc = 300, 5 MHz) | Reaches 3GPP LDPC Base Graph 1 structural ceiling (k_max = 8,448). |
| **Multipath Channel Robustness** | **AWGN & Rayleigh Fading** validated | Ultra-sparse 2.1% pilot grid completely resolves 90-degree phase ambiguity. |
| **Model Size & Latency** | **128,021 weights** (0.488 MB memory) | Batched CPU latency of **0.98 ms** (meets 5G NR 1.0 ms slot constraint). |
| **Visual Payload Verification** | **Infinite PSNR** at 8 dB | Bit-for-bit perfect image reconstruction; 38.5 dB PSNR at 6 dB. |

---

## 📁 Repository Directory Structure

- **IEEE_Paper_Pilot_Free_Neural_Demapper_5G_NR.tex**: Master publication-ready IEEE LaTeX manuscript.
- **Master_Project_Understanding_and_Defense_Guide.pdf**: Master oral defense and theory guide (PDF).
- **Master_Project_Understanding_and_Defense_Guide.md**: Companion markdown defense guide.
- **Joint_Neural_Demod_FEC/**: Core Production Codebase (Pipeline, Models, Benchmarks, Evaluations, Paper).
- **
eports AND results/**: All Formal PDF Reports and Deliverables.
- **PROFESSOR REVIEW 1/**: Professor Review Tasks Implementation (Multipath, TDL, Complexity, Unit Tests).
- **TDL_64QAM_Extension/**: Advanced Channel & Modulation Extensions.

---

## 🚀 Quickstart & Execution

`ash
# Run Main Baseline Experiment
python Joint_Neural_Demod_FEC/run_experiment.py --fft_size 64 --channel awgn

# 1. Delta-Loop Ablation Study
python Joint_Neural_Demod_FEC/eval_delta_loop_comprehensive.py

# 2. Pilot Removal & Spectral Efficiency Study
python Joint_Neural_Demod_FEC/eval_pilot_ablation_comprehensive.py

# 3. Computational Complexity & Latency Profiling
python Joint_Neural_Demod_FEC/eval_complexity_comprehensive.py

# 4. End-to-End Image Transmission Verification
python Joint_Neural_Demod_FEC/transmit_image_demo.py
`
