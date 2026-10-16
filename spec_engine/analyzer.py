"""
Theoretical speedup modeling and empirical telemetry for speculative decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class SpeedupMetrics:
    """Theoretical and empirical performance characteristics."""
    gamma: int
    acceptance_rate: float
    draft_cost_ratio: float
    expected_accepted_tokens: float
    speedup_factor: float
    effective_compression: float


class SpeculativeSpeedupAnalyzer:
    """Computes speedup models and tracks empirical decoding throughput."""

    def __init__(self) -> None:
        self.empirical_accepted_counts: List[int] = []

    def calculate_theoretical_speedup(
        self,
        gamma: int,
        acceptance_rate: float,
        draft_cost_ratio: float = 0.05,
    ) -> SpeedupMetrics:
        """
        Calculates theoretical speedup using Leviathan et al. formula:
        E[Tokens] = (1 - alpha^(gamma + 1)) / (1 - alpha)
        Speedup = E[Tokens] / (gamma * c + 1)
        where c = cost_draft / cost_target.
        """
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1, received {gamma}")
        if not (0.0 <= acceptance_rate <= 1.0):
            raise ValueError(f"acceptance_rate must be in [0, 1], received {acceptance_rate}")

        alpha = acceptance_rate

        if math.isclose(alpha, 1.0):
            expected_tokens = float(gamma + 1)
        elif math.isclose(alpha, 0.0):
            expected_tokens = 1.0
        else:
            expected_tokens = (1.0 - (alpha ** (gamma + 1))) / (1.0 - alpha)

        # Total step cost relative to 1 target forward pass
        step_cost = (gamma * draft_cost_ratio) + 1.0
        speedup = expected_tokens / step_cost
        compression = expected_tokens / float(gamma + 1)

        return SpeedupMetrics(
            gamma=gamma,
            acceptance_rate=alpha,
            draft_cost_ratio=draft_cost_ratio,
            expected_accepted_tokens=round(expected_tokens, 4),
            speedup_factor=round(speedup, 4),
            effective_compression=round(compression, 4),
        )

    def record_step(self, accepted_count: int) -> None:
        """Records the number of tokens emitted in a single verification step."""
        self.empirical_accepted_counts.append(accepted_count)

    def get_empirical_summary(self, draft_cost_ratio: float = 0.05, gamma: int = 5) -> dict:
        """Computes summary statistics across all recorded steps."""
        if not self.empirical_accepted_counts:
            return {"total_steps": 0, "avg_tokens_per_step": 0.0, "empirical_speedup": 0.0}

        total_steps = len(self.empirical_accepted_counts)
        total_tokens = sum(self.empirical_accepted_counts)
        avg_tokens = total_tokens / float(total_steps)

        step_cost = (gamma * draft_cost_ratio) + 1.0
        empirical_speedup = avg_tokens / step_cost

        return {
            "total_steps": total_steps,
            "total_tokens_emitted": total_tokens,
            "avg_tokens_per_step": round(avg_tokens, 3),
            "empirical_speedup": round(empirical_speedup, 3),
        }
