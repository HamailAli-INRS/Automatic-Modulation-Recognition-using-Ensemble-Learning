"""
Unit Tests for Joint Neural Demodulation and FEC Pipeline.
"""

import unittest
import torch

from Joint_Neural_Demod_FEC.models.neural_demapper import ResCNNDemapper
from Joint_Neural_Demod_FEC.models.replay_buffer import ExperienceReplayBuffer
from Joint_Neural_Demod_FEC.pipeline.ofdm_fec_chain import OFDMFECPipeline
from Joint_Neural_Demod_FEC.benchmarks.classical_baseline import ClassicalPilotBaseline
from Joint_Neural_Demod_FEC.pipeline.trainer import JointDemodFECTrainer

class TestJointDemodFEC(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = OFDMFECPipeline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=4,
            coderate=0.5,
            channel_type="awgn",
            device=self.device
        )
        self.demapper = ResCNNDemapper(
            num_bits_per_symbol=4,
            hidden_dim=32,
            num_res_blocks=2
        ).to(self.device)
        self.buffer = ExperienceReplayBuffer(capacity=64)

    def test_pipeline_transmission(self):
        batch_size = 4
        u, c, y_grid, no = self.pipeline.transmit(batch_size, ebno_db=12.0)
        self.assertEqual(u.shape, (batch_size, self.pipeline.k))
        self.assertEqual(c.shape, (batch_size, self.pipeline.n))
        self.assertEqual(y_grid.shape, (batch_size, 2, 14, 64))
        self.assertGreater(no, 0.0)

    def test_neural_demapper_forward(self):
        batch_size = 4
        _, _, y_grid, _ = self.pipeline.transmit(batch_size, ebno_db=15.0)
        llrs = self.demapper(y_grid)
        self.assertEqual(llrs.shape, (batch_size, self.pipeline.n))

    def test_replay_buffer_and_trainer_step(self):
        trainer = JointDemodFECTrainer(
            self.demapper,
            self.pipeline,
            self.buffer,
            lr=1e-3,
            buffer_batch_size=8,
            device=self.device
        )
        _, _, y_grid, _ = self.pipeline.transmit(batch_size=8, ebno_db=15.0)
        metrics = trainer.train_step(y_grid)
        self.assertIn("instant_loss", metrics)
        self.assertIn("buffer_loss", metrics)
        self.assertGreater(metrics["buffer_size"], 0)

    def test_classical_baseline(self):
        baseline = ClassicalPilotBaseline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=4,
            coderate=0.5,
            channel_type="awgn",
            device=self.device
        )
        raw_ber, post_fec_ber, bler = baseline.evaluate(batch_size=8, ebno_db=15.0)
        self.assertIsInstance(raw_ber, float)
        self.assertIsInstance(post_fec_ber, float)
        self.assertIsInstance(bler, float)

if __name__ == "__main__":
    unittest.main()
