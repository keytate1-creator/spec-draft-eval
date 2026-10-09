"""
Rejection sampling and residual distribution recovery for speculative decoding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class VerificationDecision:
    """Outcome of a single speculative token verification."""
    accepted: bool
    draft_token: int
    output_token: int
    acceptance_ratio: float
    p_prob: float
    q_prob: float


class RejectionSampler:
    """
    Executes lossless rejection sampling to match the target model distribution.
    Guarantees: Output distribution is identically distributed to target model P.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def sample_from_distribution(self, dist: Dict[int, float]) -> int:
        """Samples a discrete token from a normalized probability distribution dictionary."""
        if not dist:
            raise ValueError("Cannot sample from an empty distribution.")

        r = self.rng.random()
        cumulative = 0.0
        last_token = next(iter(dist.keys()))

        for token, prob in dist.items():
            cumulative += prob
            last_token = token
            if r <= cumulative:
                return token

        return last_token

    def verify_token(
        self,
        draft_token: int,
        p_dist: Dict[int, float],
        q_dist: Dict[int, float],
    ) -> VerificationDecision:
        """
        Applies rejection sampling criterion on draft_token:
        Accept with probability min(1.0, p(x) / q(x)).
        If rejected, sample recovery token from residual distribution max(0, p(x) - q(x)).
        """
        p_prob = p_dist.get(draft_token, 0.0)
        q_prob = q_dist.get(draft_token, 0.0)

        ratio = (p_prob / q_prob) if q_prob > 0.0 else 0.0
        acc_prob = min(1.0, ratio)

        u = self.rng.random()

        if u <= acc_prob:
            # Accepted candidate
            return VerificationDecision(
                accepted=True,
                draft_token=draft_token,
                output_token=draft_token,
                acceptance_ratio=ratio,
                p_prob=p_prob,
                q_prob=q_prob,
            )

        # Rejected candidate: sample from normalized residual distribution
        residual_dist: Dict[int, float] = {}
        residual_sum = 0.0

        all_tokens = set(p_dist.keys()).union(q_dist.keys())
        for tok in all_tokens:
            diff = max(0.0, p_dist.get(tok, 0.0) - q_dist.get(tok, 0.0))
            if diff > 0.0:
                residual_dist[tok] = diff
                residual_sum += diff

        if residual_sum > 0.0:
            # Normalize residual
            normalized_residual = {
                t: w / residual_sum for t, w in residual_dist.items()
            }
            resampled_token = self.sample_from_distribution(normalized_residual)
        else:
            # Fallback directly to target distribution
            resampled_token = self.sample_from_distribution(p_dist)

        return VerificationDecision(
            accepted=False,
            draft_token=draft_token,
            output_token=resampled_token,
            acceptance_ratio=ratio,
            p_prob=p_prob,
            q_prob=q_prob,
        )
