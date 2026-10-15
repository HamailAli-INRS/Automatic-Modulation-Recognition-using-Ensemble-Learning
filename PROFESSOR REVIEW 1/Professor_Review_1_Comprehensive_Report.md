# 5G NR Physical Layer Neural Demapping & LDPC Decoding
## Technical Review Report 1: Multipath Channel Scaling, Self-Supervised Delta Loop, Pilot Ablation & Complexity Analysis

---

### Executive Summary
This document presents the complete findings addressing the committee review items:
1. **Scaled Information Block Lengths**: Tested $k = 1792$, $k = 3584$, and $k = 7168$ info bits under standard 5G NR LDPC Base Graph 1/2.
2. **Multipath Channel Models**: Replaced AWGN with **Rayleigh Multipath Fading** and **3GPP TR 38.901 TDL Ray-Tracing** channels at 36 km/h vehicular mobility and 30 ns delay spread.
3. **Quantification of Self-Supervised Delta Loop**: Benchmarked the decision-directed $\Delta$-feedback loop with replay buffer against frozen/non-adapted models and supervised upper-bounds.
4. **Isolated Impact of Neural Demapper (With vs Without Pilots)**: Compared 14.3% DMRS pilot overhead vs zero-pilot operation (+16.7% useful throughput gain).
5. **Computational Complexity Profiling**: Theoretical FLOPs, parameter counts, latency (ms/frame), and throughput (Mbps) across all block lengths.

---

### Key Numerical Results

#### 1. Computational Complexity Scaling Table
| Info Bits ($k$) | Coded Bits ($n$) | Grid ($N_{sc} \times N_{sym}$) | Parameters | ResCNN GFLOPs | Batched Latency | Effective Throughput |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1792** | 3584 | 64 $\times$ 14 | 225,989 | 0.4042 GFLOPs | 1.84 ms | **1.0 Mbps** |
| **3584** | 7168 | 128 $\times$ 14 | 225,989 | 0.8083 GFLOPs | 4.36 ms | **0.8 Mbps** |
| **7168** | 14336 | 256 $\times$ 14 | 225,989 | 1.6167 GFLOPs | 8.03 ms | **0.9 Mbps** |

---

### Generated Publication Figures
1. **Figure 1: Multipath Channel Scaling** (`fig1_multipath_channel_scaling.png`)
2. **Figure 2: Impact of Self-Supervised Delta Loop** (`fig2_self_supervised_delta_impact.png`)
3. **Figure 3: Isolated Pilot Impact & Spectral Efficiency** (`fig3_isolated_pilot_impact.png`)
4. **Figure 4: Computational Complexity & Latency Scaling** (`fig4_computational_complexity_scaling.png`)
