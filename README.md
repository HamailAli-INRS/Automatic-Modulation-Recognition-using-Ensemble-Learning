# 5G NR Joint Neural Demodulator & FEC Pipeline

An AI-native physical layer receiver for **5G NR OFDM systems** that eliminates pilot overhead via a **2D Residual CNN Demapper** trained through **decision-directed self-supervised learning** with 5G NR LDPC codes.

---

## ?? Research Overview & Motivation
Classical 5G NR systems allocate \%\text{--}15\%$ of resource elements to pilot symbols for channel estimation (e.g., Least Squares / LMMSE). This project implements an end-to-end differentiable physical layer receiver that performs blind joint channel estimation and soft Log-Likelihood Ratio (LLR) demapping without any pilot overhead.

### Key Achievements:
- **+16.7% Spectral Efficiency Gain:** \%$ of resource elements carry information data (\%$ pilot overhead).
- **2 dB Power/Coding Gain:** Reaches .0\%$ Post-FEC Bit Error Rate (BER) at \text{ dB } E_b/N_0$ (classical baseline requires \text{ dB}$).
- **Self-Supervised Online Adaptation:** Uses the 5G NR LDPC Belief Propagation decoder as an error-correcting oracle ($\Delta$-loop) combined with an Experience Replay Buffer to eliminate catastrophic forgetting.

---

## ??? Repository Structure

\\\
Joint_Neural_Demod_FEC/
+-- pipeline/
¦   +-- ofdm_fec_chain.py       # Task 1: 5G NR Pilot-Free OFDM + LDPC Pipeline
¦   +-- trainer.py              # Task 4: Self-Supervised Decision-Directed Online Trainer
+-- benchmarks/
¦   +-- classical_baseline.py   # Task 2: Standard 3GPP Pilot-Assisted (LS/LMMSE/APP) Baseline
+-- models/
¦   +-- neural_demapper.py      # Task 3: 2D ResCNN Neural Demapper Architecture
¦   +-- replay_buffer.py        # Task 4: Circular Experience Replay Buffer
+-- tests/
¦   +-- test_components.py     # End-to-end component unit tests
+-- results/
¦   +-- ber_bler_comparison.png      # BER & BLER curves
¦   +-- spectral_efficiency_gain.png # Spectral efficiency comparison
¦   +-- online_adaptation_loss.png   # Self-supervised training loss
¦   +-- experiment_results.json      # Raw simulation metrics
¦   +-- neural_demapper_weights.pt   # Pretrained model weights
+-- run_experiment.py           # Full automated benchmark & evaluation script
+-- plot_results.py             # Result plotting script
\\\

---

## ?? System Architecture

### 1. Transmitter Chain
1. **Information Bits:** Generates binary vector  \in \{0, 1\}^k$ ( = 1792$).
2. **5G NR LDPC Encoder:** Rate-/2$ encoding $\rightarrow$ codeword  \in \{0, 1\}^n$ ( = 3584$).
3. **16-QAM Mapper:** Gray-mapped constellation $\rightarrow 896$ complex symbols (\text{ bits/sym}$).
4. **OFDM Resource Grid:** Maps symbols into a \text{ symbols} \times 64\text{ subcarriers}$ grid (\text{ REs}$, zero pilots).
5. **Channel:** Rayleigh multipath block fading + AWGN noise.

### 2. Receiver Chain (Neural Path)
1. **FFT & Grid Extraction:** Received time-frequency matrix converted to Real/Imag channels $[B, 2, 14, 64]$.
2. **2D ResCNN Demapper:** Jointly learns spatial time-frequency fading correlation and outputs soft LLRs.
3. **5G NR LDPC Decoder:** 20-iteration Belief Propagation recovers the transmitted information bits $\hat{u}$.

### 3. Self-Supervised Online Delta-Loop
- **Inference:** Received grid $\rightarrow$ Raw LLRs $\rightarrow$ LDPC decoder outputs error-corrected codeword $\hat{c}$.
- **Supervision:** The corrected codeword $\hat{c}$ serves as pseudo-ground-truth.
- **Reliability Filter:** Discards frames with $>30\%$ raw bit disagreement.
- **Experience Replay:** Mini-batches sampled from a circular buffer of capacity 256 for stable AdamW updates.

---

## ?? Benchmark Results

| /N_0$ (dB) | Classical Raw BER | Classical Post-FEC BER | Neural Raw BER | Neural Post-FEC BER | Neural BLER |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0 dB** | 28.2% | 18.5% | 24.8% | 14.9% | 94.5% |
| **2 dB** | 21.6% | 10.2% | 17.5% | 5.8% | 74.2% |
| **4 dB** | 14.8% | 3.1% | 11.7% | 0.9% | 44.0% |
| **6 dB** | 8.1% | 0.4% | 5.2% | **0.0%** | **0.0%** |
| **8 dB** | 3.9% | **0.0%** | 2.1% | **0.0%** | **0.0%** |
| **10 dB** | 1.2% | **0.0%** | 0.4% | **0.0%** | **0.0%** |

---

## ?? Quickstart

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- NVIDIA Sionna 0.18+
- NumPy, Matplotlib, ReportLab

### Run Tests
\\\ash
python -m unittest Joint_Neural_Demod_FEC/tests/test_components.py
\\\

### Run Full Pipeline & Benchmark
\\\ash
python Joint_Neural_Demod_FEC/run_experiment.py
\\\

---

## ?? Technical Report
For full mathematical derivations, 3GPP LDPC Tanner graph explanations, LLR calibration equations, and line-by-line code annotations, refer to:
- [\Complete_4Task_Technical_Report.pdf\](Complete_4Task_Technical_Report.pdf)
