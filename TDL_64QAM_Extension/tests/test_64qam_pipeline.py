"""
Unit Tests for 64-QAM 3GPP TDL Physical Layer Pipeline.
"""

import unittest
import torch

from TDL_64QAM_Extension.models.neural_64qam_demapper import Neural64QAMDemapper
from TDL_64QAM_Extension.models.replay_buffer_64qam import ExperienceReplayBuffer64QAM
from TDL_64QAM_Extension.pipeline.ofdm_64qam_tdl_chain import OFDM64QAMTDLPipeline
from TDL_64QAM_Extension.benchmarks.classical_64qam_baseline import Classical64QAMPilotBaseline
from TDL_64QAM_Extension.pipeline.trainer_64qam import JointTrainer64QAM

class Test64QAMTDLPipeline(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.pipeline = OFDM64QAMTDLPipeline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=6,
            coderate=0.5,
            device=self.device
        )
        self.demapper = Neural64QAMDemapper(
            num_bits_per_symbol=6,
            hidden_dim=32,
            num_res_blocks=2
        ).to(self.device)
        self.buffer = ExperienceReplayBuffer64QAM(capacity=32)

    def test_pipeline_transmission(self):
        batch_size = 2
        u, c, y_grid, no = self.pipeline.transmit(batch_size, ebno_db=16.0)
        self.assertEqual(u.shape, (batch_size, 2688))
        self.assertEqual(c.shape, (batch_size, 5376))
        self.assertEqual(y_grid.shape, (batch_size, 2, 14, 64))
        self.assertGreater(no, 0.0)

    def test_neural_demapper_forward(self):
        batch_size = 2
        _, _, y_grid, _ = self.pipeline.transmit(batch_size, ebno_db=16.0)
        llrs = self.demapper(y_grid)
        self.assertEqual(llrs.shape, (batch_size, 5376))

    def test_classical_baseline(self):
        baseline = Classical64QAMPilotBaseline(
            num_ofdm_symbols=14,
            fft_size=64,
            num_bits_per_symbol=6,
            coderate=0.5,
            device=self.device
        )
        raw_ber, post_fec_ber, bler = baseline.evaluate(batch_size=4, ebno_db=18.0)
        self.assertIsInstance(raw_ber, float)
        self.assertIsInstance(post_fec_ber, float)
        self.assertIsInstance(bler, float)

if __name__ == "__main__":
    unittest.main()
