"""
Quick script to benchmark Rayleigh performance across all 4 block lengths:
k=1792 (fft=64), k=3584 (fft=128), k=7168 (fft=256), k=8400 (fft=300)
"""
import os
import sys
import torch
import json
import numpy as np

project_root = r"c:\Users\hamhamai001\Desktop\Amr using EL"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Joint_Neural_Demod_FEC.benchmarks.classical_baseline import ClassicalPilotBaseline

def benchmark_rayleigh():
    configs = [
        {"name": "k=1792 (64 SC)", "fft_size": 64},
        {"name": "k=3584 (128 SC)", "fft_size": 128},
        {"name": "k=7168 (256 SC)", "fft_size": 256},
        {"name": "k=8400 (300 SC)", "fft_size": 300},
    ]
    ebno_list = [6, 8, 10, 12, 14, 16, 18]
    all_results = {}
    
    for cfg in configs:
        fft = cfg["fft_size"]
        print(f"Benchmarking Classical Rayleigh for fft_size={fft}...")
        baseline = ClassicalPilotBaseline(
            num_ofdm_symbols=14,
            fft_size=fft,
            num_bits_per_symbol=4,
            coderate=0.5,
            num_ldpc_iter=20,
            channel_type="rayleigh",
            device="cpu"
        )
        res_cfg = {"classical_post_fec": [], "classical_bler": []}
        for ebno in ebno_list:
            _, fec, bler = baseline.evaluate(batch_size=16, ebno_db=float(ebno))
            res_cfg["classical_post_fec"].append(fec)
            res_cfg["classical_bler"].append(bler)
            print(f"  fft={fft} | Eb/N0={ebno} dB -> Classical Post-FEC: {fec:.5e} | BLER: {bler:.4f}")
        all_results[str(fft)] = res_cfg

    with open(os.path.join(project_root, "Joint_Neural_Demod_FEC", "results", "rayleigh_multiblock_classical.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    benchmark_rayleigh()
