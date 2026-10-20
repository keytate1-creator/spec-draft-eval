"""
Unit tests for rejection sampling and residual distribution recovery using unittest.
"""

import unittest
from spec_engine.sampler import RejectionSampler


class TestRejectionSampler(unittest.TestCase):

    def test_sample_from_distribution_deterministic(self):
        sampler = RejectionSampler(seed=42)
        dist = {101: 0.0, 102: 1.0, 103: 0.0}
        # Only token 102 has probability 1.0
        token = sampler.sample_from_distribution(dist)
        self.assertEqual(token, 102)

    def test_rejection_sampling_definite_accept(self):
        sampler = RejectionSampler(seed=42)
        # When p(x) >= q(x), acceptance ratio >= 1.0, must accept with 100% certainty
        p = {101: 0.8, 102: 0.2}
        q = {101: 0.4, 102: 0.6}

        decision = sampler.verify_token(draft_token=101, p_dist=p, q_dist=q)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.output_token, 101)
        self.assertAlmostEqual(decision.acceptance_ratio, 2.0, places=4)

    def test_rejection_sampling_definite_reject_and_recovery(self):
        sampler = RejectionSampler(seed=42)
        # When p(x) == 0, acceptance ratio is 0.0, must reject with 100% certainty
        p = {101: 0.0, 102: 1.0}
        q = {101: 1.0, 102: 0.0}

        decision = sampler.verify_token(draft_token=101, p_dist=p, q_dist=q)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.draft_token, 101)
        # Recovery token must be 102 (from target distribution residual)
        self.assertEqual(decision.output_token, 102)


if __name__ == "__main__":
    unittest.main()
