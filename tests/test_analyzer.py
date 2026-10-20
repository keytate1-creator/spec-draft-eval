"""
Unit tests for theoretical speedup calculations and telemetry using unittest.
"""

import unittest
from spec_engine.analyzer import SpeculativeSpeedupAnalyzer


class TestSpeculativeAnalyzer(unittest.TestCase):

    def test_theoretical_speedup_bounds(self):
        analyzer = SpeculativeSpeedupAnalyzer()

        # If alpha == 1.0 (perfect agreement), E[tokens] = gamma + 1
        m_perf = analyzer.calculate_theoretical_speedup(gamma=4, acceptance_rate=1.0, draft_cost_ratio=0.05)
        self.assertEqual(m_perf.expected_accepted_tokens, 5.0)

        # If alpha == 0.0 (total mismatch), E[tokens] = 1.0 (only the recovery token emitted)
        m_zero = analyzer.calculate_theoretical_speedup(gamma=4, acceptance_rate=0.0, draft_cost_ratio=0.05)
        self.assertEqual(m_zero.expected_accepted_tokens, 1.0)

        # Standard operating regime: alpha = 0.70, gamma = 3
        # E[tokens] = (1 - 0.7^4) / (1 - 0.7) = (1 - 0.2401) / 0.3 = 0.7599 / 0.3 = 2.533
        m_std = analyzer.calculate_theoretical_speedup(gamma=3, acceptance_rate=0.70, draft_cost_ratio=0.05)
        self.assertAlmostEqual(m_std.expected_accepted_tokens, 2.533, places=3)
        self.assertGreater(m_std.speedup_factor, 1.0)

    def test_empirical_telemetry(self):
        analyzer = SpeculativeSpeedupAnalyzer()
        analyzer.record_step(3)
        analyzer.record_step(4)
        analyzer.record_step(2)

        summary = analyzer.get_empirical_summary(draft_cost_ratio=0.05, gamma=4)
        self.assertEqual(summary["total_steps"], 3)
        self.assertEqual(summary["total_tokens_emitted"], 9)
        self.assertEqual(summary["avg_tokens_per_step"], 3.0)


if __name__ == "__main__":
    unittest.main()
