"""
Comprehensive Computational Complexity Multi-Condition Benchmark Script.
Evaluates 2D ResCNN Neural Demapper vs Classical Pilot-Assisted Pipeline across:
1. Information Bit Scaling (k=1792, 3584, 7168, 8400 bits)
2. Channel Models (AWGN vs Rayleigh Fading)

Metrics measured:
- Parameter count (trainable / total)
- Analytical FLOPs (multiply-accumulate operations)
- Inference latency (single frame, batched)
- End-to-end pipeline latency (including LDPC encode/decode)
- Throughput (Mbps)
- Peak memory footprint
- Training step overhead (delta loop cost)
- Classical pipeline comparison (LS + LMMSE + APP + LDPC)
"""

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = r"c:\Users\hamhamai001\Desktop\Amr using EL"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Joint_Neural_Demod_FEC.models.neural_demapper import ResCNNDemapper
from Joint_Neural_Demod_FEC.models.replay_buffer import ExperienceReplayBuffer
from Joint_Neural_Demod_FEC.pipeline.ofdm_fec_chain import OFDMFECPipeline
from Joint_Neural_Demod_FEC.pipeline.trainer import JointDemodFECTrainer
from Joint_Neural_Demod_FEC.benchmarks.classical_baseline import ClassicalPilotBaseline


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def calculate_rescnn_flops(hidden_dim, num_res_blocks, num_bits_per_symbol, num_ofdm_symbols, fft_size):
    """
    Analytical FLOPs for 2D ResCNN demapper (multiply-accumulate × 2).
    Each Conv2d(C_in, C_out, K, K) with output H×W has:
        MACs = C_in × C_out × K × K × H × W
    """
    H = num_ofdm_symbols  # 14
    W = fft_size
    K = 3  # kernel size

    # input_proj: Conv2d(2, hidden_dim, 3x3)
    macs_input = 2 * hidden_dim * K * K * H * W

    # Each ResBlock has 2 × Conv2d(hidden_dim, hidden_dim, 3x3)
    macs_per_resblock = 2 * (hidden_dim * hidden_dim * K * K * H * W)
    macs_resblocks = num_res_blocks * macs_per_resblock

    # output_conv: Conv2d(hidden_dim, num_bits_per_symbol, 3x3)
    macs_output = hidden_dim * num_bits_per_symbol * K * K * H * W

    # GroupNorm + GELU: ~3 ops per element per layer (negligible vs conv, but count)
    # input_proj: hidden_dim * H * W * 3 (norm+act)
    # ResBlocks: num_res_blocks * 2 * hidden_dim * H * W * 3
    # Total activation ops
    activation_ops = (1 + 2 * num_res_blocks) * hidden_dim * H * W * 3

    total_macs = macs_input + macs_resblocks + macs_output
    total_flops = 2 * total_macs + activation_ops  # FLOPs ≈ 2 × MACs + activations

    return total_macs, total_flops


def calculate_classical_flops(fft_size, num_ofdm_symbols, num_bits_per_symbol, num_pilot_symbols):
    """
    Analytical FLOPs for Classical Pilot-Assisted Pipeline:
    LS Channel Estimation + NN Interpolation + LMMSE Equalization + Max-Log APP Demapping.
    """
    total_res = num_ofdm_symbols * fft_size
    m_order = 2 ** num_bits_per_symbol  # 16 for 16-QAM

    # 1. LS Channel Estimation: H_LS = Y_p / X_p → complex division per pilot RE
    #    Complex division ≈ 6 real FLOPs per pilot
    ls_flops = 6 * num_pilot_symbols

    # 2. Nearest-Neighbour Interpolation across time-frequency grid
    #    ~20 FLOPs per RE (copy + lookup + distance)
    interp_flops = 20 * total_res

    # 3. LMMSE Equalization per Resource Element
    #    x_hat = H* / (|H|^2 + sigma^2) × y → ~30 real FLOPs per RE
    lmmse_flops = 30 * total_res

    # 4. Max-Log APP Demapper per data RE
    #    For each bit plane: compute distance to all 2^m constellation points
    #    FLOPs per RE = 4 × m_order × num_bits_per_symbol
    num_data_res = total_res - num_pilot_symbols
    app_flops = 4 * m_order * num_bits_per_symbol * num_data_res

    total_flops = ls_flops + interp_flops + lmmse_flops + app_flops
    return {
        "ls_estimation": ls_flops,
        "interpolation": interp_flops,
        "lmmse_equalization": lmmse_flops,
        "app_demapping": app_flops,
        "total_flops": total_flops
    }


def measure_neural_latency(model, input_shape, num_warmup=5, num_runs=20):
    """Measure inference latency for neural demapper (single and batched)."""
    device = next(model.parameters()).device

    # Single frame latency
    dummy_single = torch.randn(1, *input_shape[1:], device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_single)

    latencies_single = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_single)
            t1 = time.perf_counter()
            latencies_single.append((t1 - t0) * 1000.0)

    # Batched latency (batch=16)
    dummy_batch = torch.randn(16, *input_shape[1:], device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_batch)

    latencies_batch = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_batch)
            t1 = time.perf_counter()
            latencies_batch.append((t1 - t0) * 1000.0 / 16.0)

    return {
        "single_mean_ms": float(np.mean(latencies_single)),
        "single_std_ms": float(np.std(latencies_single)),
        "single_min_ms": float(np.min(latencies_single)),
        "single_max_ms": float(np.max(latencies_single)),
        "single_p95_ms": float(np.percentile(latencies_single, 95)),
        "batched_mean_ms": float(np.mean(latencies_batch)),
        "batched_std_ms": float(np.std(latencies_batch)),
    }


def measure_classical_latency(classical_baseline, batch_size=4, ebno_db=12.0, num_warmup=2, num_runs=5):
    """Measure end-to-end classical pipeline latency."""
    for _ in range(num_warmup):
        classical_baseline.evaluate(batch_size=batch_size, ebno_db=ebno_db)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        classical_baseline.evaluate(batch_size=batch_size, ebno_db=ebno_db)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0 / batch_size)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
    }


def measure_neural_e2e_latency(trainer, pipeline, batch_size=4, ebno_db=12.0, num_warmup=2, num_runs=5):
    """Measure end-to-end neural pipeline latency (demapper + LDPC decode)."""
    for _ in range(num_warmup):
        trainer.evaluate(batch_size=batch_size, ebno_db=ebno_db)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        trainer.evaluate(batch_size=batch_size, ebno_db=ebno_db)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0 / batch_size)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
    }


def measure_training_step_latency(trainer, pipeline, batch_size=4, num_warmup=2, num_runs=5):
    """Measure delta loop training step latency (adaptation overhead)."""
    for _ in range(num_warmup):
        _, _, y, _ = pipeline.transmit(batch_size, 12.0)
        trainer.train_step(y, 12.0)

    latencies = []
    for _ in range(num_runs):
        _, _, y, _ = pipeline.transmit(batch_size, 12.0)
        t0 = time.perf_counter()
        trainer.train_step(y, 12.0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
    }


def measure_memory_footprint(model, input_shape):
    """Measure model memory footprint."""
    # Model size in bytes
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    # Activation memory estimate for single frame
    dummy = torch.randn(1, *input_shape[1:])
    activation_elements = 0
    hooks = []

    def hook_fn(module, input, output):
        nonlocal activation_elements
        if isinstance(output, torch.Tensor):
            activation_elements += output.numel()

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.GroupNorm)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model.eval()
        _ = model(dummy)

    for h in hooks:
        h.remove()

    activation_bytes = activation_elements * 4  # float32

    return {
        "param_bytes": param_bytes,
        "param_mb": param_bytes / (1024 * 1024),
        "buffer_bytes": buffer_bytes,
        "activation_bytes_single": activation_bytes,
        "activation_mb_single": activation_bytes / (1024 * 1024),
        "total_inference_mb": (param_bytes + activation_bytes) / (1024 * 1024),
    }


def run_complexity_benchmark(fft_size, channel_type, hidden_dim=48, num_res_blocks=3, num_bits_per_symbol=4):
    """Run full complexity benchmark for a single configuration."""
    print(f"\n{'='*70}")
    print(f"  Benchmarking: fft_size={fft_size}, channel={channel_type}")
    print(f"{'='*70}")

    device = "cpu"
    num_ofdm_symbols = 14

    # Create neural demapper
    model = ResCNNDemapper(
        num_bits_per_symbol=num_bits_per_symbol,
        hidden_dim=hidden_dim,
        num_res_blocks=num_res_blocks
    ).to(device)

    # Create pipeline
    pipeline = OFDMFECPipeline(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=0.5,
        num_ldpc_iter=20,
        channel_type=channel_type,
        device=device
    )

    # Create classical baseline
    classical = ClassicalPilotBaseline(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=0.5,
        num_ldpc_iter=20,
        channel_type=channel_type,
        device=device
    )

    # Create trainer for training step measurement
    replay_buffer = ExperienceReplayBuffer(capacity=64)
    trainer = JointDemodFECTrainer(model, pipeline, replay_buffer, lr=1e-3, device=device)
    # Quick warmup to have valid buffer
    trainer.bootstrap_warmup(steps=2, batch_size=4)

    # 1. Parameter Count
    trainable_params, total_params = count_parameters(model)
    print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # 2. FLOPs
    macs, flops = calculate_rescnn_flops(hidden_dim, num_res_blocks, num_bits_per_symbol, num_ofdm_symbols, fft_size)
    classical_flops_detail = calculate_classical_flops(fft_size, num_ofdm_symbols, num_bits_per_symbol, classical.num_pilot_symbols)
    print(f"  Neural FLOPs: {flops:,} ({flops/1e6:.2f} MFLOPs)")
    print(f"  Classical FLOPs: {classical_flops_detail['total_flops']:,} ({classical_flops_detail['total_flops']/1e6:.2f} MFLOPs)")

    # 3. Neural Demapper Latency
    input_shape = (1, 2, num_ofdm_symbols, fft_size)
    neural_latency = measure_neural_latency(model, input_shape)
    print(f"  Neural Inference Latency: {neural_latency['single_mean_ms']:.2f} ms (single), {neural_latency['batched_mean_ms']:.2f} ms (batched)")

    # 4. Classical Pipeline Latency
    classical_latency = measure_classical_latency(classical, batch_size=16)
    print(f"  Classical Pipeline Latency: {classical_latency['mean_ms']:.2f} ms/frame")

    # 5. End-to-End Neural Pipeline Latency
    neural_e2e = measure_neural_e2e_latency(trainer, pipeline, batch_size=16)
    print(f"  Neural E2E Latency: {neural_e2e['mean_ms']:.2f} ms/frame")

    # 6. Training Step Latency (Delta Loop Overhead)
    train_latency = measure_training_step_latency(trainer, pipeline, batch_size=16, num_warmup=3, num_runs=20)
    print(f"  Delta Loop Training Step: {train_latency['mean_ms']:.2f} ms/step")

    # 7. Memory Footprint
    memory = measure_memory_footprint(model, input_shape)
    print(f"  Model Memory: {memory['param_mb']:.3f} MB params, {memory['total_inference_mb']:.3f} MB total inference")

    # 8. Throughput
    k = pipeline.k
    throughput_single = (k / (neural_latency['single_mean_ms'] * 1e-3)) / 1e6
    throughput_batched = (k / (neural_latency['batched_mean_ms'] * 1e-3)) / 1e6
    classical_throughput = (classical.k / (classical_latency['mean_ms'] * 1e-3)) / 1e6
    print(f"  Neural Throughput: {throughput_single:.2f} Mbps (single), {throughput_batched:.2f} Mbps (batched)")
    print(f"  Classical Throughput: {classical_throughput:.2f} Mbps")

    return {
        "fft_size": fft_size,
        "channel_type": channel_type,
        "num_ofdm_symbols": num_ofdm_symbols,
        "info_bits_k_neural": pipeline.k,
        "coded_bits_n_neural": pipeline.n,
        "info_bits_k_classical": classical.k,
        "coded_bits_n_classical": classical.n,
        "grid_dimensions": f"{num_ofdm_symbols} x {fft_size}",
        "modulation": f"{2**num_bits_per_symbol}-QAM",
        "hidden_dim": hidden_dim,
        "num_res_blocks": num_res_blocks,
        # Parameters
        "trainable_params": trainable_params,
        "total_params": total_params,
        # FLOPs
        "neural_macs": macs,
        "neural_flops": flops,
        "neural_gflops": flops / 1e9,
        "neural_mflops": flops / 1e6,
        "classical_flops_detail": classical_flops_detail,
        "classical_flops": classical_flops_detail["total_flops"],
        "classical_mflops": classical_flops_detail["total_flops"] / 1e6,
        "flops_ratio": flops / max(classical_flops_detail["total_flops"], 1),
        # Latency
        "neural_latency": neural_latency,
        "classical_latency": classical_latency,
        "neural_e2e_latency": neural_e2e,
        "training_step_latency": train_latency,
        # Memory
        "memory": memory,
        # Throughput
        "neural_throughput_single_mbps": throughput_single,
        "neural_throughput_batched_mbps": throughput_batched,
        "classical_throughput_mbps": classical_throughput,
        # Pilot overhead
        "classical_pilot_overhead_pct": classical.pilot_overhead * 100,
        "neural_pilot_overhead_pct": 0.0,
    }


def generate_plots(all_results, out_dir):
    """Generate 3 publication-quality figures."""

    # Extract AWGN results for main comparison
    awgn_results = [r for r in all_results if r["channel_type"] == "awgn"]
    rayleigh_results = [r for r in all_results if r["channel_type"] == "rayleigh"]

    k_values = [r["info_bits_k_neural"] for r in awgn_results]
    fft_values = [r["fft_size"] for r in awgn_results]

    # ─── Figure 1: FLOPs Comparison (Neural vs Classical) ───
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(k_values))
    width = 0.35

    neural_mflops = [r["neural_mflops"] for r in awgn_results]
    classical_mflops = [r["classical_mflops"] for r in awgn_results]

    bars1 = ax.bar(x - width/2, neural_mflops, width, label="2D ResCNN Neural Demapper", color='#2563eb', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, classical_mflops, width, label="Classical (LS+LMMSE+APP)", color='#dc2626', edgecolor='white', linewidth=0.5)

    ax.set_xlabel("Information Bits (k)", fontsize=12, fontweight='bold')
    ax.set_ylabel("FLOPs (MFLOPs)", fontsize=12, fontweight='bold')
    ax.set_title("Computational Complexity: Neural Demapper vs Classical Pipeline", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k:,}\n(FFT={f})" for k, f in zip(k_values, fft_values)])
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + max(neural_mflops)*0.01, f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + max(neural_mflops)*0.01, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_flops_comparison.png"))
    plt.close()
    print("[OK] Figure 1: FLOPs Comparison saved.")

    # ─── Figure 2: Latency & Throughput Comparison ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Left: Latency
    neural_lat_single = [r["neural_latency"]["single_mean_ms"] for r in awgn_results]
    neural_lat_batch = [r["neural_latency"]["batched_mean_ms"] for r in awgn_results]
    classical_lat = [r["classical_latency"]["mean_ms"] for r in awgn_results]
    neural_e2e_lat = [r["neural_e2e_latency"]["mean_ms"] for r in awgn_results]

    ax1.plot(k_values, neural_lat_single, 'b-o', label="Neural (Single Frame)", linewidth=2, markersize=8)
    ax1.plot(k_values, neural_lat_batch, 'b--s', label="Neural (Batched, /32)", linewidth=2, markersize=8)
    ax1.plot(k_values, neural_e2e_lat, 'g-^', label="Neural E2E (+ LDPC)", linewidth=2, markersize=8)
    ax1.plot(k_values, classical_lat, 'r-D', label="Classical E2E (LS+LMMSE+APP+LDPC)", linewidth=2, markersize=8)

    ax1.set_xlabel("Information Bits (k)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Latency (ms / frame)", fontsize=11, fontweight='bold')
    ax1.set_title("Inference Latency Scaling", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Right: Throughput
    neural_tput_single = [r["neural_throughput_single_mbps"] for r in awgn_results]
    neural_tput_batch = [r["neural_throughput_batched_mbps"] for r in awgn_results]
    classical_tput = [r["classical_throughput_mbps"] for r in awgn_results]

    ax2.bar(np.arange(len(k_values)) - 0.25, neural_tput_single, 0.25, label="Neural (Single)", color='#3b82f6')
    ax2.bar(np.arange(len(k_values)), neural_tput_batch, 0.25, label="Neural (Batched)", color='#1d4ed8')
    ax2.bar(np.arange(len(k_values)) + 0.25, classical_tput, 0.25, label="Classical", color='#ef4444')

    ax2.set_xlabel("Information Bits (k)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Throughput (Mbps)", fontsize=11, fontweight='bold')
    ax2.set_title("Throughput Scaling", fontsize=12, fontweight='bold')
    ax2.set_xticks(np.arange(len(k_values)))
    ax2.set_xticklabels([f"k={k:,}" for k in k_values])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_latency_throughput.png"))
    plt.close()
    print("[OK] Figure 2: Latency & Throughput saved.")

    # ─── Figure 3: AWGN vs Rayleigh Overhead + Training Step Cost ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Left: AWGN vs Rayleigh Classical Latency
    if rayleigh_results:
        awgn_classical = [r["classical_latency"]["mean_ms"] for r in awgn_results]
        rayleigh_classical = [r["classical_latency"]["mean_ms"] for r in rayleigh_results]
        awgn_neural = [r["neural_e2e_latency"]["mean_ms"] for r in awgn_results]
        rayleigh_neural = [r["neural_e2e_latency"]["mean_ms"] for r in rayleigh_results]

        x = np.arange(len(k_values))
        width = 0.2
        ax1.bar(x - 1.5*width, awgn_classical, width, label="Classical AWGN", color='#fca5a5', edgecolor='#dc2626')
        ax1.bar(x - 0.5*width, rayleigh_classical, width, label="Classical Rayleigh", color='#dc2626', edgecolor='#991b1b')
        ax1.bar(x + 0.5*width, awgn_neural, width, label="Neural AWGN", color='#93c5fd', edgecolor='#2563eb')
        ax1.bar(x + 1.5*width, rayleigh_neural, width, label="Neural Rayleigh", color='#2563eb', edgecolor='#1e40af')

        ax1.set_xlabel("Information Bits (k)", fontsize=11, fontweight='bold')
        ax1.set_ylabel("E2E Latency (ms / frame)", fontsize=11, fontweight='bold')
        ax1.set_title("AWGN vs Rayleigh: Channel Impact on Latency", fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"k={k:,}" for k in k_values])
        ax1.legend(fontsize=8)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Right: Training Step (Delta Loop) Overhead
    train_awgn = [r["training_step_latency"]["mean_ms"] for r in awgn_results]
    infer_awgn = [r["neural_e2e_latency"]["mean_ms"] for r in awgn_results]
    overhead_pct = [(t / max(i, 0.01) - 1) * 100 for t, i in zip(train_awgn, infer_awgn)]

    ax2_twin = ax2.twinx()
    bars = ax2.bar(np.arange(len(k_values)) - 0.15, train_awgn, 0.3, label="Training Step (ms)", color='#f59e0b', edgecolor='#d97706')
    ax2.bar(np.arange(len(k_values)) + 0.15, infer_awgn, 0.3, label="Inference Only (ms)", color='#10b981', edgecolor='#059669')
    ax2_twin.plot(np.arange(len(k_values)), overhead_pct, 'r--D', label="Adaptation Overhead (%)", linewidth=2, markersize=8)

    ax2.set_xlabel("Information Bits (k)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Latency (ms)", fontsize=11, fontweight='bold')
    ax2_twin.set_ylabel("Adaptation Overhead (%)", color='red', fontsize=11, fontweight='bold')
    ax2.set_title("Delta Loop Training Overhead vs Inference", fontsize=12, fontweight='bold')
    ax2.set_xticks(np.arange(len(k_values)))
    ax2.set_xticklabels([f"k={k:,}" for k in k_values])
    ax2.legend(loc='upper left', fontsize=9)
    ax2_twin.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_channel_comparison_training_overhead.png"))
    plt.close()
    print("[OK] Figure 3: Channel Comparison & Training Overhead saved.")


def main():
    out_dir = os.path.join(project_root, "Joint_Neural_Demod_FEC", "results", "complexity_study")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    # Run all configurations: 4 fft_sizes × 2 channel types
    configs = [
        (64, "awgn"), (128, "awgn"), (256, "awgn"), (300, "awgn"),
        (64, "rayleigh"), (128, "rayleigh"), (256, "rayleigh"), (300, "rayleigh"),
    ]

    for fft_size, channel in configs:
        result = run_complexity_benchmark(fft_size, channel, hidden_dim=48, num_res_blocks=3)
        all_results.append(result)

    # Save JSON
    json_path = os.path.join(out_dir, "complexity_comprehensive_study.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    print(f"\n[OK] Full Complexity Study JSON saved to: {json_path}")

    # Generate plots
    generate_plots(all_results, out_dir)

    # Print summary table
    print(f"\n{'='*120}")
    print(f"{'Config':^20} | {'Params':^10} | {'Neural MFLOPs':^14} | {'Classical MFLOPs':^16} | {'Ratio':^8} | {'Latency (ms)':^14} | {'Throughput (Mbps)':^18}")
    print(f"{'='*120}")
    for r in all_results:
        config = f"k={r['info_bits_k_neural']:,} {r['channel_type'].upper()}"
        params = f"{r['trainable_params']:,}"
        n_mf = f"{r['neural_mflops']:.1f}"
        c_mf = f"{r['classical_mflops']:.1f}"
        ratio = f"{r['flops_ratio']:.0f}x"
        lat = f"{r['neural_latency']['batched_mean_ms']:.2f}"
        tput = f"{r['neural_throughput_batched_mbps']:.2f}"
        print(f"{config:^20} | {params:^10} | {n_mf:^14} | {c_mf:^16} | {ratio:^8} | {lat:^14} | {tput:^18}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
