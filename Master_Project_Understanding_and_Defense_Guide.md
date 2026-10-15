# Master Project Understanding & Oral Defense Guide
## 5G NR Joint Neural Demodulation and LDPC Decoding: Theory, Equations & Defense Script

---

### Executive Overview & Key Specs

| Parameter / Metric | Specification / Value |
|:---|:---|
| **Core Architecture** | Pilot-Free 2D ResCNN Demapper + 5G NR LDPC Belief Propagation Decoder |
| **Modulation & Code** | 16-QAM ($M=4$ bits/symbol), 3GPP LDPC Base Graph 1 ($R=0.5$) |
| **Model Size** | **128,021 trainable weights** (0.488 MB in float32) |
| **Spectral Efficiency Gain** | **+16.67%** ($2.000$ vs. $1.714	ext{ bits/s/Hz}$) |
| **Coding Advantage** | **+2.0 dB to +3.8 dB** waterfall gain over classical 3GPP baseline |
| **Inference Latency** | **0.98 ms / frame** (batched CPU, within 1.0 ms 5G slot timing) |
| **Maximum Block Length** | $k = 8,400$ bits ($N_{	ext{sc}} = 300$, 5 MHz channel, 3GPP LDPC structural ceiling) |
| **Channel Conditions** | Additive White Gaussian Noise (AWGN) & Dispersive Rayleigh Multipath Fading |

---

## Chapter 1: The Core Story in Plain English (The 2-Minute Pitch)

1. **The Real-World Wireless Problem (The Pilot Tax):**
   In standardized 5G New Radio (NR), whenever a mobile phone or base station transmits data across radio waves, it is forced to waste **14.29% of its total bandwidth** on fixed benchmark signals called *Pilots* (Demodulation Reference Signals, DMRS). Pilots carry zero user information—they exist solely so the receiver can estimate how the wireless channel distorted the signal. Worse yet, in low-signal environments (low SNR), estimating the channel using noisy pilots causes severe errors that corrupt the entire packet.

2. **Our Invention (Pilot-Free Joint Neural Demapping):**
   We completely eliminate all pilots (100% data transmission), delivering an immediate, permanent **+16.67% spectral efficiency gain**. Instead of relying on fragile channel estimation formulas, we deploy a lightweight **2D Residual Convolutional Neural Network (2D ResCNN)**. The CNN processes the entire time-frequency resource grid directly like a 2D image, capturing complex spatial noise correlations and outputting soft bit probabilities.

3. **The Self-Supervised Magic (The Teacher-Student Delta Loop):**
   How can the CNN learn online without any pilots or ground-truth bit labels? We pair the CNN with a standardized **5G NR LDPC Belief Propagation Decoder**. Think of the CNN as an *eager student* and the LDPC decoder as a *strict mathematical spell-checker*:
   - The student (CNN) makes an initial guess of the bits (which contains 10%–20% raw errors).
   - The LDPC decoder uses strict parity-check geometry ($\mathbf{H} \cdot \mathbf{c}^T = \mathbf{0}$) to correct these errors, producing certified mathematical truth.
   - We compute the difference (the **Delta signal $\Delta$**) between what the CNN guessed and what the LDPC decoder proved.
   - The CNN updates its weights using this Delta signal through an Experience Replay Buffer—**learning continuously online without true labels!**

---

## Chapter 2: The End-to-End Pipeline Step-by-Step

- **Stage 1 (TX - Payload):** Information bits $\mathbf{u} \in \{0, 1\}^k$ are generated ($k = 1,792$ up to $8,400$ bits).
- **Stage 2 (TX - FEC Encode):** 5G NR LDPC BG1 encodes $\mathbf{u}$ into codeword $\mathbf{c}$ of length $n = k/R$ ($n = 3,584$ to $16,800$ bits).
- **Stage 3 (TX - Modulation):** 16-QAM maps codeword $\mathbf{c}$ into complex symbols $\mathbf{s}$ (4 bits/symbol).
- **Stage 4 (TX - Grid Mapping):** Mapped into a $14 	imes N_{	ext{sc}}$ Resource Grid with **100% data and 0% pilots**. IFFT + CP generates time-domain signal.
- **Stage 5 (Channel):** Propagates across AWGN or Rayleigh multipath fading with noise power $N_0$.
- **Stage 6 (RX - Grid Reconstruction):** CP removal + FFT reconstructs the 2D grid $\mathbf{Y}$ formatted as 2 real channels $[\Re(\mathbf{Y}), \Im(\mathbf{Y})]$.
- **Stage 7 (RX - Neural Demapping):** 2D ResCNN outputs calibrated soft Log-Likelihood Ratios (LLRs) $\hat{oldsymbol{\lambda}}$.
- **Stage 8 (RX - LDPC Decoding):** Belief Propagation decoder runs 20 message-passing iterations to output user bits $\hat{\mathbf{u}}$.
- **Stage 9 (Adaptation - Delta Loop):** Re-encodes $\hat{\mathbf{c}} = 	ext{LDPC-Encode}(\hat{\mathbf{u}})$. If disagreement $d \le 30\%$, pushes to Replay Buffer and updates weights via AdamW.

---

## Chapter 3: Key Mathematical Equations Decoded

- **Equation 1 (Parity Geometry):** $\mathbf{H}_{	ext{LDPC}} \mathbf{c}^T = \mathbf{0} \pmod 2$. Proves codeword validity without external labels.
- **Equation 2 (Received Signal):** $Y[l, k] = H[l, k] X[l, k] + W[l, k]$.
- **Equation 3 (Noise PSD):** $N_0 = rac{E_s}{M \cdot R \cdot 10^{0.1 \cdot (E_b/N_0)_{	ext{dB}}}}$. Converts SNR in dB to exact noise variance.
- **Equation 4 (Rayleigh Envelope):** $p(|H|) = rac{|H|}{\sigma^2} e^{-|H|^2 / (2\sigma^2)}$.
- **Equation 5 (Pilot Overhead):** $lpha_{	ext{pilot}} = rac{2}{14} pprox 14.29\%$ in classical 3GPP Kronecker patterns.
- **Equations 7–10 (Spectral Efficiency Gain):** $\eta_{	ext{Classical}} = 1.714	ext{ bits/s/Hz}$ vs. $\eta_{	ext{Proposed}} = 2.000	ext{ bits/s/Hz}$ $	o$ **$+16.67\%$ Net Gain**.
- **Equation 17 (Parameter Count):** Exactly **128,021 parameters** ($0.488	ext{ MB}$ memory footprint).
- **Equation 20 (Reliability Disagreement):** $d = rac{1}{n} \sum |	ext{Hard}(\hat{\lambda}) - \hat{c}|$. Replay buffer gating threshold set to $	au = 0.30$.
- **Equation 22 (Rotational Ambiguity):** $\mathbb{E}[(e^{j\pi/2} s)^4] = \mathbb{E}[s^4]$. Explains why pure blind demapping requires an ultra-sparse 2.1% pilot phase anchor.

---

## Chapter 4: Professor Defense Script (Top 7 Tough Questions)

### Q1: Why does the classical baseline fail at 6 dB while your neural receiver gets zero errors?
> **Answer:** At 6 dB, Least Squares channel estimation on pilots suffers from noise enhancement. The noisy estimates distort LMMSE equalization, causing the classical APP demapper to produce compressed, miscalibrated LLRs that fail the LDPC decoder on 65% of frames ($	ext{BLER}=0.65$). The 2D ResCNN looks at the entire grid jointly, performing spatial denoising across adjacent subcarriers and symbols to produce sharp LLRs (91.37% raw accuracy), easily achieving $	ext{BER}=0.00000$.

### Q2: Isn't your self-supervised Delta loop just an artifact of good initial weights?
> **Answer:** No. Our 4-way ablation (Table V) proves that freezing weights (Delta OFF) leaves a permanent error floor ($	ext{BER}=8.45 	imes 10^{-2}$ at 6 dB). Turning the Delta loop ON reduces demapper-decoder disagreement from 26.8% to 0.1%, yielding a $+3.8	ext{ dB}$ gain and operating within $0.2	ext{ dB}$ of the ideal supervised upper bound without ground-truth labels.

### Q3: Does this scale to real 5G NR block lengths beyond toy simulations?
> **Answer:** Yes. We evaluated block lengths up to $k = 8,400$ bits ($N_{	ext{sc}} = 300$, 5 MHz channel), which operates at the structural ceiling of 3GPP LDPC Base Graph 1 ($k_{\max} = 8,448$). While classical BLER degrades at large block lengths, the 2D ResCNN maintains $	ext{BER}=0.00000$ across all block sizes.

### Q4: Why did pure blind neural demapping fail in Rayleigh fading, and what is your solution?
> **Answer:** Pure blind demapping encounters the fundamental $90^\circ$ rotational symmetry of square QAM constellations, where $\mathbb{E}[(e^{j\pi/2}s)^4] = \mathbb{E}[s^4]$, causing $	ext{BER}pprox 0.49$. We resolved this with an ultra-sparse pilot grid of only 2 subcarriers per slot (2.1% overhead), which anchors the phase and achieves $	ext{BER}=0.00000$ while preserving over 95% of our throughput advantage.

### Q5: Is this practical for real-time edge hardware?
> **Answer:** Yes. The model has only 128,021 parameters ($0.488	ext{ MB}$ footprint) and runs in **$0.98	ext{ ms}$ per frame** on CPU, satisfying the 5G NR $1.0	ext{ ms}$ slot constraint. The online training step runs in a background thread, adding zero latency to packet forwarding.

---

## Chapter 5: Key Numbers Cheat Sheet to Memorize

- **Spectral Efficiency Gain:** $+16.67\%$ ($2.000$ vs. $1.714	ext{ bits/s/Hz}$)
- **Coding Gain:** $+2.0	ext{ dB}$ to $+3.8	ext{ dB}$ waterfall advantage over 3GPP baseline
- **Model Size:** $128,021$ parameters ($0.488	ext{ MB}$ memory footprint)
- **CPU Inference Latency:** $0.98	ext{ ms}$ (batched, 5G slot compliant)
- **Max Block Size:** $k = 8,400$ bits ($N_{	ext{sc}} = 300$, 3GPP LDPC BG1 ceiling)
- **Disagreement Drop:** $26.8\% 	o 0.1\%$ via self-supervised Delta loop
- **Phase Anchor:** $2.1\%$ sparse pilot resolves $90^\circ$ rotational ambiguity in Rayleigh fading
- **Image Payload Recovery:** Infinite PSNR at $8	ext{ dB}$, near-perfect ($38.5	ext{ dB}$ PSNR) at $6	ext{ dB}$
