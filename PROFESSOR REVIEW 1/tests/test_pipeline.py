"""
Unit and Integration Tests for Professor Review 1 Architecture and Pipelines.
"""

import unittest
import torch
import os
import sys

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.rescnn_2d_demapper import ResCNNDemapper2D
from pipeline.ofdm_channel_pipeline import OFDMChannelPipeline
from pipeline.self_supervised_trainer import SelfSupervisedTrainer
from pipeline.replay_buffer import ExperienceReplayBuffer
from benchmarks.complexity_profiler import ComplexityProfiler

class TestProfessorReview1Pipeline(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_rescnn_dimensions_and_flops(self):
        """Test ResCNN forward pass across multiple grid sizes."""
        for n_sc in [64, 128, 256]:
            model = ResCNNDemapper2D(num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3).to(self.device)
            dummy_input = torch.randn(2, 2, 14, n_sc, device=self.device)
            out_llrs = model(dummy_input)
            expected_bits = n_sc * 14 * 4
            self.assertEqual(out_llrs.shape, (2, expected_bits))
            
            flops = model.calculate_flops(num_subcarriers=n_sc, num_ofdm_symbols=14)
            self.assertGreater(flops, 0)
            self.assertGreater(model.count_parameters(), 10000)

    def test_ofdm_channels(self):
        """Test AWGN, Rayleigh, and TDL channels."""
        for ch in ["awgn", "rayleigh", "tdl_raytracing"]:
            pipe = OFDMChannelPipeline(
                num_ofdm_symbols=14,
                fft_size=64,
                num_bits_per_symbol=4,
                coderate=0.5,
                channel_type=ch,
                pilot_mode="pilot_free",
                device=self.device
            )
            u, c, y_grid, no, rx_raw = pipe.transmit(batch_size=4, ebno_db=12.0)
            self.assertEqual(u.shape, (4, 1792))
            self.assertEqual(c.shape, (4, 3584))
            self.assertEqual(y_grid.shape, (4, 2, 14, 64))

    def test_self_supervised_step(self):
        """Test one step of self-supervised delta adaptation."""
        pipe = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=4,
            coderate=0.5,
            channel_type="rayleigh",
            pilot_mode="pilot_free",
            device=self.device
        )
        model = ResCNNDemapper2D(num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3).to(self.device)
        trainer = SelfSupervisedTrainer(model, pipe, device=self.device)
        
        _, _, y_grid, _, _ = pipe.transmit(batch_size=4, ebno_db=14.0)
        res = trainer.self_supervised_delta_step(y_grid)
        self.assertIn("loss", res)
        self.assertIn("disagreement", res)

    def test_pilot_assisted_pipeline(self):
        """Test pilot-assisted pipeline and demapping."""
        pipe = OFDMChannelPipeline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=4,
            coderate=0.5,
            channel_type="rayleigh",
            pilot_mode="pilot_assisted",
            device=self.device
        )
        model = ResCNNDemapper2D(num_bits_per_symbol=4, hidden_dim=64, num_res_blocks=3).to(self.device)
        trainer = SelfSupervisedTrainer(model, pipe, device=self.device)
        u, c, y_grid, no, rx_raw = pipe.transmit(batch_size=4, ebno_db=12.0)
        self.assertEqual(c.shape, (4, 3072))
        llrs = model(y_grid, data_mask=pipe.data_mask)
        self.assertEqual(llrs.shape, (4, 3072))
        trainer.supervised_train(steps=2, batch_size=4, verbose=False)

    def test_complexity_profiler(self):
        """Test computational complexity profiling."""
        profiler = ComplexityProfiler(device=self.device)
        res = profiler.profile_rescnn(num_subcarriers_list=[64, 128], num_runs=5)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["info_bits_k"], 1792)
        self.assertEqual(res[1]["info_bits_k"], 3584)

if __name__ == "__main__":
    unittest.main()
