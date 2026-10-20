# SpecDraft: Speculative Decoding Rejection-Sampling & Verification Engine for LLMs

[![CI](https://github.com/keytate1-creator/spec-draft-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/keytate1-creator/spec-draft-eval/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Flake8](https://img.shields.io/badge/code%20style-flake8-000000.svg)](https://flake8.pycqa.org/)

`SpecDraft` is an open-source, zero-dependency engine implementing **provably lossless Speculative Decoding** (Leviathan et al., Chen et al.) and **Tree-Based Speculative Verification** (Medusa, Eagle) for Large Language Model acceleration.

Speculative decoding overcomes the memory bandwidth bottleneck of auto-regressive decoding by drafting $K$ candidate tokens with an inexpensive model and verifying them in parallel with a target model. `SpecDraft` provides exact rejection sampling, residual distribution recovery, tree-structured hypothesis verification, and latency speedup modeling.

---

## Mathematical Specification

### 1. Rejection Sampling Criterion (Lossless Guarantee)

Given a target model probability distribution $p(x)$ and draft model distribution $q(x)$ over vocabulary $V$:

$$\text{Acceptance Probability: } \alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

If candidate token $x_i$ is rejected, it is resampled from the normalized residual distribution:

$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y \in V} \max(0, p(y) - q(y))}$$

This guarantees that the output distribution is **mathematically identical** to sampling directly from the target model ($P_{\text{spec}}(x) \equiv P_{\text{target}}(x)$).

---

### 2. Tree-Structured Speculation (Medusa / Eagle)

Instead of a linear chain of $K$ tokens, `SpecDraft` evaluates candidate tokens arranged in a directed tree $T = (V_T, E_T)$. The target model computes logits for all candidate nodes in a single forward pass via a 2D attention mask. The engine identifies and accepts the longest valid path where all nodes satisfy rejection sampling.

```
                         Root (Last Generated Token)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
                Token A1                              Token A2
             (P_acc = 0.92)                        (P_acc = 0.41)
                   │                                     │
         ┌─────────┴─────────┐                           ▼
         ▼                   ▼                        Token B3
      Token B1            Token B2                 (P_acc = 0.12)
   (P_acc = 0.85)      (P_acc = 0.28)
         │
         ▼
      Token C1
   (P_acc = 0.79)  ──> Accepted Path: [A1, B1, C1] (3 tokens in 1 step!)
```

---

## Installation

```bash
git clone https://github.com/keytate1-creator/spec-draft-eval.git
cd spec-draft-eval
pip install -e .
```

---

## Programmatic Usage

```python
from spec_engine.sampler import RejectionSampler
from spec_engine.analyzer import SpeculativeSpeedupAnalyzer

# Initialize sampler
sampler = RejectionSampler(seed=42)

# Simulated target distribution p and draft distribution q over vocabulary
vocab_size = 1000
p = {101: 0.70, 102: 0.20, 103: 0.10}
q = {101: 0.60, 102: 0.30, 103: 0.10}

# Verify drafted token 101
decision = sampler.verify_token(draft_token=101, p_dist=p, q_dist=q)

print(f"Token Accepted: {decision.accepted}")
print(f"Acceptance Ratio: {decision.acceptance_ratio:.2f}")
print(f"Final Output Token: {decision.output_token}")

# Model theoretical speedup
analyzer = SpeculativeSpeedupAnalyzer()
metrics = analyzer.calculate_theoretical_speedup(
    gamma=5,               # Draft 5 tokens per step
    acceptance_rate=0.75,  # 75% mean agreement
    draft_cost_ratio=0.08  # Draft model is 12.5x faster than target
)

print(f"Expected Accepted Tokens Per Step: {metrics.expected_accepted_tokens:.2f}")
print(f"Wall-Clock Speedup Factor: {metrics.speedup_factor:.2f}x")
```

---

## CLI Simulation

Run a Monte Carlo simulation comparing linear chain vs. tree-based speculative decoding:

```bash
python -m spec_engine.cli simulate --steps 1000 --gamma 5 --vocab-size 5000 --agreement 0.80
```

---

## License

MIT License. Designed and maintained by Kaylor Tate Cespedes.
