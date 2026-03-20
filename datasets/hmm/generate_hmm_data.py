#!/usr/bin/env python3
"""
Generate data from Hidden Markov Models used in mechanistic interpretability.

Implements the HMM processes from:
- Shai et al. (2024) "Transformers Represent Belief State Geometry in their
  Residual Stream"
- Also related: Elhage et al., Olsson et al. on in-context learning

Processes implemented:
1. Mess3: 3-state HMM with emissions {A, B, C}
2. Z1R (Zero-One-Random): 3-state process
3. RRXOR: Random-Random-XOR process

Usage:
    python generate_hmm_data.py --process mess3 --num_sequences 1000 --seq_length 256
    python generate_hmm_data.py --process z1r --num_sequences 1000 --seq_length 256
    python generate_hmm_data.py --process rrxor --num_sequences 1000 --seq_length 256
    python generate_hmm_data.py --all --num_sequences 1000 --seq_length 256

References:
    - Shai et al. "Transformers Represent Belief State Geometry in their
      Residual Stream" (2024)
"""

import argparse
import json
from pathlib import Path

import numpy as np


# =============================================================================
# Mess3 Process
# =============================================================================
class Mess3:
    """
    The Mess3 (Messy 3-state) HMM from Shai et al. 2024.

    Hidden states: {S0, S1, S2}
    Emissions: {A=0, B=1, C=2}

    Transition matrix (rows = from, cols = to):
      Each state transitions to the other two states with probability 0.5 each
      (never self-loops).

    Emission matrix:
      S0 emits: A with p=0.9, B with p=0.05, C with p=0.05
      S1 emits: A with p=0.05, B with p=0.9, C with p=0.05
      S2 emits: A with p=0.05, B with p=0.05, C with p=0.9

    The "messy" part is that emissions are noisy -- each state has a dominant
    emission but can produce any token.
    """

    name = "mess3"
    num_states = 3
    vocab_size = 3
    token_names = ["A", "B", "C"]

    def __init__(self, emission_prob=0.9):
        self.emission_prob = emission_prob
        noise = (1.0 - emission_prob) / 2.0

        # Transition: uniform over other states
        self.transition = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])

        # Emission: dominant token per state
        self.emission = np.array([
            [emission_prob, noise, noise],
            [noise, emission_prob, noise],
            [noise, noise, emission_prob],
        ])

        # Stationary distribution is uniform
        self.initial = np.array([1/3, 1/3, 1/3])

    def generate(self, length, rng=None):
        """Generate a sequence of (hidden_states, emissions)."""
        if rng is None:
            rng = np.random.default_rng()

        states = np.zeros(length, dtype=np.int32)
        emissions = np.zeros(length, dtype=np.int32)

        # Initial state
        states[0] = rng.choice(self.num_states, p=self.initial)
        emissions[0] = rng.choice(self.vocab_size, p=self.emission[states[0]])

        for t in range(1, length):
            states[t] = rng.choice(self.num_states, p=self.transition[states[t-1]])
            emissions[t] = rng.choice(self.vocab_size, p=self.emission[states[t]])

        return states, emissions

    def get_config(self):
        return {
            "name": self.name,
            "num_states": self.num_states,
            "vocab_size": self.vocab_size,
            "emission_prob": self.emission_prob,
            "transition": self.transition.tolist(),
            "emission": self.emission.tolist(),
        }


# =============================================================================
# Z1R (Zero-One-Random) Process
# =============================================================================
class Z1R:
    """
    Zero-One-Random (Z1R) process from Shai et al. 2024.

    Hidden states: {Z, O, R} (Zero, One, Random)
    Emissions: {0, 1}

    - State Z always emits 0
    - State O always emits 1
    - State R emits 0 or 1 with equal probability

    Transitions: Cycle Z -> O -> R -> Z with some probability of staying.
    Specifically (from Shai et al.):
      Z -> O with p=0.5, Z -> R with p=0.5
      O -> R with p=0.5, O -> Z with p=0.5
      R -> Z with p=0.5, R -> O with p=0.5
    """

    name = "z1r"
    num_states = 3
    vocab_size = 2
    token_names = ["0", "1"]

    def __init__(self):
        # Transition: uniform over other states (no self-loops)
        self.transition = np.array([
            [0.0, 0.5, 0.5],  # Z -> {O, R}
            [0.5, 0.0, 0.5],  # O -> {Z, R}
            [0.5, 0.5, 0.0],  # R -> {Z, O}
        ])

        # Emission
        self.emission = np.array([
            [1.0, 0.0],  # Z always emits 0
            [0.0, 1.0],  # O always emits 1
            [0.5, 0.5],  # R emits uniformly
        ])

        self.initial = np.array([1/3, 1/3, 1/3])

    def generate(self, length, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        states = np.zeros(length, dtype=np.int32)
        emissions = np.zeros(length, dtype=np.int32)

        states[0] = rng.choice(self.num_states, p=self.initial)
        emissions[0] = rng.choice(self.vocab_size, p=self.emission[states[0]])

        for t in range(1, length):
            states[t] = rng.choice(self.num_states, p=self.transition[states[t-1]])
            emissions[t] = rng.choice(self.vocab_size, p=self.emission[states[t]])

        return states, emissions

    def get_config(self):
        return {
            "name": self.name,
            "num_states": self.num_states,
            "vocab_size": self.vocab_size,
            "transition": self.transition.tolist(),
            "emission": self.emission.tolist(),
        }


# =============================================================================
# RRXOR (Random-Random-XOR) Process
# =============================================================================
class RRXOR:
    """
    Random-Random-XOR (RRXOR) process from Shai et al. 2024.

    This is NOT a standard HMM but a deterministic process with structure:
    Tokens come in groups of 3: (a, b, a XOR b)
    where a and b are each independently 0 or 1 with equal probability.

    The sequence is: a1, b1, (a1 XOR b1), a2, b2, (a2 XOR b2), ...

    Hidden states track position within the triplet and the values seen so far.
    Vocab: {0, 1}

    For modeling as an HMM, we use 6 hidden states:
      (pos=0), (pos=1,a=0), (pos=1,a=1), (pos=2,a=0,b=0), (pos=2,a=0,b=1),
      (pos=2,a=1,b=0), (pos=2,a=1,b=1)
    But it's simpler to generate directly.
    """

    name = "rrxor"
    num_states = 3  # position in triplet (simplified view)
    vocab_size = 2
    token_names = ["0", "1"]

    def __init__(self):
        pass

    def generate(self, length, rng=None):
        """Generate RRXOR sequence.

        States encode position in triplet: 0=first(random), 1=second(random), 2=third(xor).
        """
        if rng is None:
            rng = np.random.default_rng()

        states = np.zeros(length, dtype=np.int32)
        emissions = np.zeros(length, dtype=np.int32)

        t = 0
        while t < length:
            a = rng.integers(0, 2)
            b = rng.integers(0, 2)
            xor = a ^ b

            triplet_states = [0, 1, 2]
            triplet_emissions = [a, b, xor]

            for i in range(3):
                if t + i < length:
                    states[t + i] = triplet_states[i]
                    emissions[t + i] = triplet_emissions[i]
            t += 3

        return states, emissions

    def get_config(self):
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "description": "Sequence of triplets (a, b, a XOR b) where a,b ~ Bernoulli(0.5)",
        }


# =============================================================================
# Generic HMM (for custom experiments)
# =============================================================================
class GenericHMM:
    """
    A generic HMM for controlled experiments with known ground-truth geometry.

    Useful for creating HMMs with specific properties:
    - Known belief state polytope structure
    - Controlled number of states and vocabulary size
    - Tunable emission noise
    """

    def __init__(self, transition, emission, initial=None, name="generic"):
        """
        Args:
            transition: (num_states, num_states) transition matrix
            emission: (num_states, vocab_size) emission matrix
            initial: (num_states,) initial distribution (default: uniform)
        """
        self.transition = np.array(transition, dtype=np.float64)
        self.emission = np.array(emission, dtype=np.float64)
        self.num_states = self.transition.shape[0]
        self.vocab_size = self.emission.shape[1]
        self.name = name

        if initial is None:
            self.initial = np.ones(self.num_states) / self.num_states
        else:
            self.initial = np.array(initial, dtype=np.float64)

        # Validate
        assert self.transition.shape == (self.num_states, self.num_states)
        assert np.allclose(self.transition.sum(axis=1), 1.0)
        assert np.allclose(self.emission.sum(axis=1), 1.0)

    def generate(self, length, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        states = np.zeros(length, dtype=np.int32)
        emissions = np.zeros(length, dtype=np.int32)

        states[0] = rng.choice(self.num_states, p=self.initial)
        emissions[0] = rng.choice(self.vocab_size, p=self.emission[states[0]])

        for t in range(1, length):
            states[t] = rng.choice(self.num_states, p=self.transition[states[t-1]])
            emissions[t] = rng.choice(self.vocab_size, p=self.emission[states[t]])

        return states, emissions

    def belief_update(self, belief, observation):
        """Compute exact Bayesian belief update given an observation.

        This is key for comparing transformer representations to optimal
        belief states. The belief state after observing token x is:

            b'(s') ∝ P(x|s') * Σ_s T(s,s') * b(s)

        Args:
            belief: Current belief vector (num_states,)
            observation: Observed token (int)

        Returns:
            Updated belief vector (num_states,)
        """
        # Predict: b_pred(s') = Σ_s T(s,s') * b(s)
        predicted = self.transition.T @ belief
        # Update: b'(s') ∝ P(x|s') * b_pred(s')
        updated = self.emission[:, observation] * predicted
        updated /= updated.sum()
        return updated

    def compute_belief_trajectory(self, emissions):
        """Compute full sequence of belief states for a given emission sequence.

        Args:
            emissions: Array of observed tokens.

        Returns:
            beliefs: (len(emissions)+1, num_states) array of belief states.
                     beliefs[0] is the prior, beliefs[t] is after observing emissions[t-1].
        """
        beliefs = np.zeros((len(emissions) + 1, self.num_states))
        beliefs[0] = self.initial.copy()

        for t, obs in enumerate(emissions):
            beliefs[t+1] = self.belief_update(beliefs[t], obs)

        return beliefs

    def get_config(self):
        return {
            "name": self.name,
            "num_states": self.num_states,
            "vocab_size": self.vocab_size,
            "transition": self.transition.tolist(),
            "emission": self.emission.tolist(),
            "initial": self.initial.tolist(),
        }


# =============================================================================
# Convenience factories
# =============================================================================

PROCESSES = {
    "mess3": Mess3,
    "z1r": Z1R,
    "rrxor": RRXOR,
}


def generate_dataset(process_name, num_sequences, seq_length, seed=42):
    """Generate a dataset from a named process.

    Returns:
        dict with keys:
            'emissions': (num_sequences, seq_length) int array
            'states': (num_sequences, seq_length) int array
            'config': dict with process parameters
    """
    rng = np.random.default_rng(seed)
    process_cls = PROCESSES[process_name]
    process = process_cls()

    all_states = np.zeros((num_sequences, seq_length), dtype=np.int32)
    all_emissions = np.zeros((num_sequences, seq_length), dtype=np.int32)

    for i in range(num_sequences):
        states, emissions = process.generate(seq_length, rng=rng)
        all_states[i] = states
        all_emissions[i] = emissions

    return {
        "emissions": all_emissions,
        "states": all_states,
        "config": process.get_config(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HMM datasets")
    parser.add_argument("--process", type=str, choices=list(PROCESSES.keys()),
                       help="Which process to generate")
    parser.add_argument("--all", action="store_true",
                       help="Generate all processes")
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: same as script)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    processes_to_run = list(PROCESSES.keys()) if args.all else [args.process]

    if not args.all and args.process is None:
        parser.error("Must specify --process or --all")

    for proc_name in processes_to_run:
        print(f"Generating {proc_name}: {args.num_sequences} sequences x {args.seq_length} tokens...")
        data = generate_dataset(proc_name, args.num_sequences, args.seq_length, args.seed)

        # Save
        prefix = output_dir / f"{proc_name}"
        np.save(f"{prefix}_emissions.npy", data["emissions"])
        np.save(f"{prefix}_states.npy", data["states"])

        config_path = output_dir / f"{proc_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(data["config"], f, indent=2)

        print(f"  Saved: {prefix}_emissions.npy, {prefix}_states.npy, {config_path}")
        print(f"  Emissions shape: {data['emissions'].shape}")
        print(f"  Token distribution: {np.bincount(data['emissions'].flatten())}")


if __name__ == "__main__":
    main()
