#!/usr/bin/env python3
"""
Generate synthetic HMM datasets with known ground-truth geometry for
controlled experiments on residual stream representations.

Key idea: By designing HMMs with specific properties, we can create datasets
where the optimal belief state has known geometric structure (simplices,
polytopes, etc.) and then check whether transformer residual streams learn
to represent this geometry.

Experiments included:
1. 2-state HMM with binary emissions (simplex = line segment)
2. 3-state HMM with 3 emissions (simplex = triangle)
3. K-state HMM with varying emission noise (controls belief geometry)
4. HMM with known mixed-membership structure

Usage:
    python generate_synthetic_hmm.py --experiment all --num_sequences 1000 --seq_length 256
    python generate_synthetic_hmm.py --experiment 2state --num_sequences 5000

References:
    - Shai et al. 2024 "Transformers Represent Belief State Geometry"
    - Rabiner 1989 "A Tutorial on Hidden Markov Models"
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Import the GenericHMM from the hmm module
sys.path.insert(0, str(Path(__file__).parent.parent / "hmm"))
from generate_hmm_data import GenericHMM


def make_2state_binary(noise=0.1):
    """
    2-state HMM with binary emissions.

    The belief state space is the 1-simplex (a line segment [0,1]).
    With low noise, beliefs tend to concentrate near the vertices.
    With high noise, beliefs spread across the simplex.

    Args:
        noise: Emission noise level. 0 = deterministic, 0.5 = fully random.
    """
    transition = np.array([
        [0.3, 0.7],
        [0.7, 0.3],
    ])
    emission = np.array([
        [1 - noise, noise],
        [noise, 1 - noise],
    ])
    return GenericHMM(transition, emission, name=f"2state_noise{noise}")


def make_3state_triangle(noise=0.05):
    """
    3-state HMM with 3 emissions.

    The belief state space is the 2-simplex (a triangle).
    This is the simplest case where belief geometry is 2D.
    """
    transition = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    base = noise / 2
    diag = 1 - noise
    emission = np.array([
        [diag, base, base],
        [base, diag, base],
        [base, base, diag],
    ])
    return GenericHMM(transition, emission, name=f"3state_noise{noise}")


def make_kstate(k, vocab_size=None, noise=0.05):
    """
    K-state HMM with configurable vocab size.

    The belief state space is the (K-1)-simplex.
    """
    if vocab_size is None:
        vocab_size = k

    # Uniform transition to other states
    transition = np.ones((k, k)) / (k - 1)
    np.fill_diagonal(transition, 0)

    # Emission: each state has a dominant emission
    emission = np.full((k, vocab_size), noise / (vocab_size - 1))
    for i in range(k):
        emission[i, i % vocab_size] = 1 - noise

    return GenericHMM(transition, emission, name=f"{k}state_v{vocab_size}_noise{noise}")


def make_asymmetric_2state(p_stay_0=0.8, p_stay_1=0.2, noise=0.1):
    """
    2-state HMM with asymmetric transitions.

    State 0 is "sticky" (stays with high probability).
    State 1 is "transient" (leaves quickly).

    This creates non-uniform belief dynamics: beliefs about being in
    state 0 are updated differently than beliefs about state 1.
    """
    transition = np.array([
        [p_stay_0, 1 - p_stay_0],
        [1 - p_stay_1, p_stay_1],
    ])
    emission = np.array([
        [1 - noise, noise],
        [noise, 1 - noise],
    ])
    return GenericHMM(transition, emission, name=f"asym2_ps0{p_stay_0}_ps1{p_stay_1}")


def make_noisy_clock(k=4, noise=0.05):
    """
    K-state deterministic clock with noisy emissions.

    Transitions: 0 -> 1 -> 2 -> ... -> K-1 -> 0 (deterministic cycle)
    Emissions: State i primarily emits token i, with some noise.

    The belief state has interesting temporal structure:
    after observing enough tokens, the belief should track position in the cycle.
    """
    transition = np.zeros((k, k))
    for i in range(k):
        transition[i, (i + 1) % k] = 1.0

    base = noise / (k - 1)
    emission = np.full((k, k), base)
    np.fill_diagonal(emission, 1 - noise)

    return GenericHMM(transition, emission, name=f"clock{k}_noise{noise}")


# =============================================================================
# Dataset generation with belief trajectories
# =============================================================================

def generate_with_beliefs(hmm, num_sequences, seq_length, seed=42):
    """Generate data and compute ground-truth belief trajectories.

    This is the key output for comparing with transformer residual streams:
    we want to see if the transformer's internal representations align with
    the Bayesian belief states.

    Returns:
        dict with:
            'emissions': (N, T) observed tokens
            'states': (N, T) hidden states
            'beliefs': (N, T+1, K) belief state trajectories
            'config': HMM parameters
    """
    rng = np.random.default_rng(seed)
    K = hmm.num_states

    all_emissions = np.zeros((num_sequences, seq_length), dtype=np.int32)
    all_states = np.zeros((num_sequences, seq_length), dtype=np.int32)
    all_beliefs = np.zeros((num_sequences, seq_length + 1, K), dtype=np.float32)

    for i in range(num_sequences):
        states, emissions = hmm.generate(seq_length, rng=rng)
        beliefs = hmm.compute_belief_trajectory(emissions)

        all_emissions[i] = emissions
        all_states[i] = states
        all_beliefs[i] = beliefs.astype(np.float32)

    return {
        'emissions': all_emissions,
        'states': all_states,
        'beliefs': all_beliefs,
        'config': hmm.get_config(),
    }


EXPERIMENTS = {
    "2state": lambda: make_2state_binary(noise=0.1),
    "2state_clean": lambda: make_2state_binary(noise=0.01),
    "2state_noisy": lambda: make_2state_binary(noise=0.3),
    "3state": lambda: make_3state_triangle(noise=0.05),
    "3state_noisy": lambda: make_3state_triangle(noise=0.2),
    "4state": lambda: make_kstate(4, noise=0.05),
    "5state": lambda: make_kstate(5, noise=0.05),
    "asymmetric": lambda: make_asymmetric_2state(),
    "clock4": lambda: make_noisy_clock(4, noise=0.05),
    "clock8": lambda: make_noisy_clock(8, noise=0.05),
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic HMM datasets with belief trajectories")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=list(EXPERIMENTS.keys()) + ["all"])
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    exps = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    for exp_name in exps:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")

        hmm = EXPERIMENTS[exp_name]()
        data = generate_with_beliefs(hmm, args.num_sequences, args.seq_length, args.seed)

        prefix = output_dir / exp_name
        np.save(f"{prefix}_emissions.npy", data['emissions'])
        np.save(f"{prefix}_states.npy", data['states'])
        np.save(f"{prefix}_beliefs.npy", data['beliefs'])

        with open(f"{prefix}_config.json", "w") as f:
            json.dump(data['config'], f, indent=2)

        print(f"  HMM: {hmm.num_states} states, {hmm.vocab_size} tokens")
        print(f"  Data: {data['emissions'].shape}")
        print(f"  Beliefs: {data['beliefs'].shape}")
        print(f"  Saved to: {prefix}_*.npy")

        # Print some statistics
        belief_var = data['beliefs'][:, 1:, :].var(axis=(0, 1))
        print(f"  Belief variance per state: {belief_var}")


if __name__ == "__main__":
    main()
