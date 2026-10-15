"""
Comprehensive Computational Complexity Profiler for 2D ResCNN Neural Demapper and Classical Baseline.
Profiles FLOPs, Parameter counts, Inference Latencies (ms/frame), Throughput (Mbps), and Memory Footprint
across multiple information bit block lengths: k=1792 (Nsc=64), k=3584 (Nsc=128), k=7168 (Nsc=256).
"""

import time
import torch
import numpy as np
import sys
import os

try:
    from ..models.rescnn_2d_demapper import ResCNNDemapper2D
except Exception:
    from models.rescnn_2d_demapper import ResCNNDemapper2D

class ComplexityProfiler:
    """
    Evaluates theoretical and empirical computational complexity metrics.
    """
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def profile_rescnn(self, num_subcarriers_list=(64, 128, 256), num_ofdm_symbols=14, num_bits_per_symbol=4, num_runs=50):
        """
        Profiles 2D ResCNN across multiple resource grid sizes and bit lengths.
        """
        results = []
        for n_sc in num_subcarriers_list:
            total_symbols = n_sc * num_ofdm_symbols
            total_coded_bits = total_symbols * num_bits_per_symbol
            info_bits_k = int(total_coded_bits * 0.5) # Rate 1/2
            
            model = ResCNNDemapper2D(
                num_bits_per_symbol=num_bits_per_symbol,
                hidden_dim=64,
                num_res_blocks=3
            ).to(self.device)
            model.eval()
            
            # 1. Parameter Count
            params = model.count_parameters()
            
            # 2. Analytical FLOPs / MACs per frame
            flops_per_frame = model.calculate_flops(num_subcarriers=n_sc, num_ofdm_symbols=num_ofdm_symbols)
            gflops = flops_per_frame / 1e9
            
            # 3. Dummy input tensor [1, 2, Symbols, Subcarriers]
            dummy_input = torch.randn(1, 2, num_ofdm_symbols, n_sc, device=self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
                    
            # 4. Latency timing (single frame)
            latencies_single = []
            with torch.no_grad():
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    _ = model(dummy_input)
                    t1 = time.perf_counter()
                    latencies_single.append((t1 - t0) * 1000.0) # ms
                    
            mean_latency_single = float(np.mean(latencies_single))
            std_latency_single = float(np.std(latencies_single))
            
            # 5. Batched Latency (Batch=32)
            dummy_batch = torch.randn(32, 2, num_ofdm_symbols, n_sc, device=self.device)
            latencies_batch = []
            with torch.no_grad():
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    _ = model(dummy_batch)
                    t1 = time.perf_counter()
                    latencies_batch.append((t1 - t0) * 1000.0 / 32.0) # ms per frame in batch
                    
            mean_latency_batch = float(np.mean(latencies_batch))
            
            # Throughput (Mbps) = (Info Bits) / (Latency in seconds)
            throughput_single_mbps = (info_bits_k / (mean_latency_single * 1e-3)) / 1e6
            throughput_batched_mbps = (info_bits_k / (mean_latency_batch * 1e-3)) / 1e6
            
            # 6. Classical LMMSE + APP Demapper Theoretical Complexity
            # LMMSE Channel Inversion: O(N_sc * N_ant^3) + interpolation O(N_sc * N_sym)
            # APP Demapper per resource element: O(N_sc * N_sym * 2^M) where 2^M = 16 for 16-QAM
            # Approx FLOPs for Classical LMMSE + APP:
            m_order = 2 ** num_bits_per_symbol # 16 for 16-QAM, 64 for 64-QAM
            classical_flops_per_frame = (
                # LS pilot est: 6 * N_pilots
                6 * (n_sc * 2) +
                # 2D interpolation: ~20 FLOPs per RE
                20 * total_symbols +
                # LMMSE equalization per RE: ~30 FLOPs per RE
                30 * total_symbols +
                # Max-Log APP Demapper per RE: ~4 * m_order * num_bits_per_symbol
                4 * m_order * num_bits_per_symbol * total_symbols
            )
            classical_gflops = classical_flops_per_frame / 1e9
            
            results.append({
                "subcarriers": n_sc,
                "ofdm_symbols": num_ofdm_symbols,
                "modulation": f"{2**num_bits_per_symbol}-QAM",
                "coded_bits_n": total_coded_bits,
                "info_bits_k": info_bits_k,
                "model_parameters": params,
                "rescnn_flops_per_frame": flops_per_frame,
                "rescnn_gflops": gflops,
                "classical_gflops": classical_gflops,
                "latency_single_frame_ms": mean_latency_single,
                "latency_single_std_ms": std_latency_single,
                "latency_batched_frame_ms": mean_latency_batch,
                "throughput_single_mbps": throughput_single_mbps,
                "throughput_batched_mbps": throughput_batched_mbps
            })
            
        return results

    def print_complexity_table(self, results):
        """Print clean formatted table of computational complexity."""
        print("\n" + "=" * 92)
        print(f"{'Info Bits (k)':^13} | {'Coded (n)':^11} | {'Params':^10} | {'ResCNN GFLOPs':^14} | {'Latency (ms)':^14} | {'Throughput (Mbps)':^18}")
        print("=" * 92)
        for r in results:
            k = r["info_bits_k"]
            n = r["coded_bits_n"]
            params = f"{r['model_parameters']:,}"
            gflops = f"{r['rescnn_gflops']:.4f}"
            lat = f"{r['latency_batched_frame_ms']:.2f} ms"
            tput = f"{r['throughput_batched_mbps']:.2f} Mbps"
            print(f"{k:^13d} | {n:^11d} | {params:^10s} | {gflops:^14s} | {lat:^14s} | {tput:^18s}")
        print("=" * 92 + "\n")
