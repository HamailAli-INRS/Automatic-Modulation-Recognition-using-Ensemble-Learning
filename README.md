# 5G NR Joint Neural Demodulation & LDPC Decoding via Decision-Directed Self-Supervised Learning

An AI-native physical layer receiver for **5G New Radio (NR) and 6G OFDM systems** that eliminates pilot overhead via a **2D Residual CNN Demapper (2D ResCNN)** trained through **decision-directed self-supervised learning** with 5G NR LDPC codes.

---

## 🎯 Executive Highlights & Key Findings

| Metric / Parameter | Value / Achievement | Practical Significance |
|---|:---:|---|
| **Spectral Efficiency Gain** | **+16.67%** (2.00 vs. 1.71 b/s/Hz for 16-QAM; 3.00 vs. 2.57 for 64-QAM) | 100% data payload, 0% pilot overhead across all block lengths. |
| **Extended Dynamic Range** | **-10 dB to +30 dB in 2 dB steps** (21 evaluated operating points) | Full waterfall curves from noise floor to high-SNR zero error. |
| **High-Order Modulation (64-QAM)** | **Zero post-FEC BER at 8.0 dB** | 2.0 dB coding gain over classical 3GPP; beats SotA by 2.0 dB. |
| **Standard Modulation (16-QAM)** | **Zero post-FEC BER at 6.0 dB** | 2.0 dB coding gain over classical 3GPP with zero pilots. |
| **State-of-the-Art Benchmarking** | **Outperforms Ye et al. (IEEE WCL) by 2.0 dB** | Solves uncoded error floors; zero label dependency. |
| **Rayleigh Fading Symmetry Fix** | **Ultra-sparse 2.1% anchor resolves 90° spin** | Recovers BER = 0.00000 at 10 dB, saving >85% of 3GPP pilot overhead. |
| **Self-Supervised Adaptation** | Disagreement dropped **26.8% -> 0.1%** | Closes within 0.2 dB of genie-aided bound with **zero true labels**. |
| **Model Size & Latency** | **128,021 weights** (0.488 MB memory) | Batched CPU latency of **0.98 ms** (meets 5G NR 1.0 ms slot constraint). |
| **Visual Payload Verification** | **Infinite PSNR** at 8 dB | Bit-for-bit perfect image reconstruction; 38.5 dB PSNR at 6 dB. |

---

## 📁 Repository Directory Structure

- **`IEEE_CommLett_PilotFree_Neural_Demapper.tex`**: Latest 4+1 IEEE Communications Letters / Conference LaTeX manuscript (includes 2020–2026 SotA benchmarks).
- **`Extended_SNR_and_High_Order_Modulation_Report.pdf`**: Comprehensive extended SNR (-10 to +30 dB), 64-QAM, and SotA comparison report (PDF).
- **`ber_master_results.png`**: Master 3-panel publication figure spanning 16-QAM AWGN, 64-QAM AWGN, and Rayleigh multipath fading.
- **`Master_Project_Understanding_and_Defense_Guide.pdf`**: Master oral defense and theory guide (PDF).
- **`Master_Project_Understanding_and_Defense_Guide.md`**: Companion markdown defense guide.
- **`Joint_Neural_Demod_FEC/`**: Core Production Codebase:
  - `pipeline/`: 5G NR OFDM + LDPC transmission chain and trainer.
  - `models/`: 2D ResCNN demapper and experience replay buffer.
  - `benchmarks/`: Classical 3GPP pilot baseline (LS + LMMSE + APP).
  - `paper/`: Complete LaTeX source, figures, and compiled assets.
  - `results/extended/`: Extended numerical JSON results and high-res curves.
- **`reports AND results/`**: All 11 formal PDF technical reports and deliverables.
- **`PROFESSOR REVIEW 1/`**: Professor review tasks implementation (Multipath, TDL, Complexity, Unit Tests).
- **`TDL_64QAM_Extension/`**: Advanced channel and modulation extensions.

---

## 🚀 Quickstart & Execution

```bash
# 1. Run Extended SNR Sweep (-10 to +30 dB) & High-Order 64-QAM
python Joint_Neural_Demod_FEC/eval_extended_snr_modulation.py --channel awgn
python Joint_Neural_Demod_FEC/eval_extended_snr_modulation.py --channel rayleigh

# 2. Run Baseline Pilot-Free Experiment (16-QAM)
python Joint_Neural_Demod_FEC/run_experiment.py --fft_size 64 --channel awgn

# 3. Delta-Loop Ablation Study
python Joint_Neural_Demod_FEC/eval_delta_loop_comprehensive.py

# 4. Pilot Removal & Spectral Efficiency Study
python Joint_Neural_Demod_FEC/eval_pilot_ablation_comprehensive.py

# 5. Computational Complexity & Latency Profiling
python Joint_Neural_Demod_FEC/eval_complexity_comprehensive.py

# 6. End-to-End Image Transmission Verification
python Joint_Neural_Demod_FEC/transmit_image_demo.py
```
