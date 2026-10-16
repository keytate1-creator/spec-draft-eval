"""
Command-line interface and Monte Carlo simulator for SpecDraft.
"""

from __future__ import annotations

import sys
import random
import argparse
from typing import Sequence, Optional, Dict

from spec_engine.sampler import RejectionSampler
from spec_engine.analyzer import SpeculativeSpeedupAnalyzer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spec-draft",
        description="Speculative Decoding Rejection-Sampling & Verification Engine for LLMs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: simulate
    sim_parser = subparsers.add_parser("simulate", help="Run Monte Carlo speculative decoding simulation.")
    sim_parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of verification steps to simulate (default: 1000)",
    )
    sim_parser.add_argument(
        "--gamma",
        type=int,
        default=5,
        help="Number of drafted candidate tokens per step (default: 5)",
    )
    sim_parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Vocabulary size (default: 1000)",
    )
    sim_parser.add_argument(
        "--agreement",
        type=float,
        default=0.75,
        help="Mean probability alignment between target and draft (default: 0.75)",
    )
    sim_parser.add_argument(
        "--draft-cost-ratio",
        type=float,
        default=0.06,
        help="Latency ratio of draft model to target model (default: 0.06)",
    )
    sim_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser


def generate_aligned_distributions(
    vocab_size: int,
    agreement: float,
    rng: random.Random,
) -> tuple[Dict[int, float], Dict[int, float]]:
    """Generates synthetic target p and draft q probability distributions with given agreement."""
    top_token = rng.randint(0, vocab_size - 1)

    p_dist: Dict[int, float] = {top_token: 0.65}
    q_dist: Dict[int, float] = {}

    if rng.random() <= agreement:
        q_dist[top_token] = 0.65
    else:
        q_dist[top_token] = 0.20

    remaining_p = 1.0 - p_dist[top_token]
    remaining_q = 1.0 - q_dist[top_token]

    # Distribute among noise tokens
    noise_count = min(10, vocab_size - 1)
    for i in range(noise_count):
        tok = (top_token + 1 + i) % vocab_size
        p_dist[tok] = remaining_p / noise_count
        q_dist[tok] = remaining_q / noise_count

    return p_dist, q_dist


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        rng = random.Random(args.seed)
        sampler = RejectionSampler(seed=args.seed)
        analyzer = SpeculativeSpeedupAnalyzer()

        theoretical = analyzer.calculate_theoretical_speedup(
            gamma=args.gamma,
            acceptance_rate=args.agreement,
            draft_cost_ratio=args.draft_cost_ratio,
        )

        for _ in range(args.steps):
            accepted_in_step = 0
            for _ in range(args.gamma):
                p_dist, q_dist = generate_aligned_distributions(
                    args.vocab_size, args.agreement, rng
                )
                draft_token = sampler.sample_from_distribution(q_dist)
                decision = sampler.verify_token(draft_token, p_dist, q_dist)
                if decision.accepted:
                    accepted_in_step += 1
                else:
                    # Emits recovery token on rejection and breaks chain
                    accepted_in_step += 1
                    break
            else:
                # If all gamma accepted, emit 1 final target token
                accepted_in_step += 1

            analyzer.record_step(accepted_in_step)

        empirical = analyzer.get_empirical_summary(
            draft_cost_ratio=args.draft_cost_ratio,
            gamma=args.gamma,
        )

        sys.stdout.write("==================================================\n")
        sys.stdout.write(" SPECULATIVE DECODING SIMULATION METRICS\n")
        sys.stdout.write("==================================================\n")
        sys.stdout.write(f" Simulated Steps         : {args.steps:,}\n")
        sys.stdout.write(f" Draft Horizon (gamma)   : {args.gamma}\n")
        sys.stdout.write(f" Agreement Probability   : {args.agreement:.2%}\n")
        sys.stdout.write(f" Draft Cost Ratio        : {args.draft_cost_ratio:.2%}\n")
        sys.stdout.write("--------------------------------------------------\n")
        sys.stdout.write(f" Theoretical Speedup     : {theoretical.speedup_factor:.2f}x\n")
        sys.stdout.write(f" Expected Tokens/Step    : {theoretical.expected_accepted_tokens:.2f}\n")
        sys.stdout.write("--------------------------------------------------\n")
        sys.stdout.write(f" Empirical Speedup       : {empirical['empirical_speedup']:.2f}x\n")
        sys.stdout.write(f" Empirical Tokens/Step   : {empirical['avg_tokens_per_step']:.2f}\n")
        sys.stdout.write(f" Total Tokens Emitted    : {empirical['total_tokens_emitted']:,}\n")
        sys.stdout.write("==================================================\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
