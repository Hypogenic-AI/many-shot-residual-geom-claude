#!/usr/bin/env python3
"""
Generate simple game sequence datasets for studying residual stream geometry.

Games implemented:
1. Rock-Paper-Scissors (RPS): Sequences of plays with various strategies
2. Tic-Tac-Toe: Complete game transcripts with random or strategic play
3. Simple Poker: Simplified poker hand sequences

These are used to study how transformers represent game state information
in their residual streams, especially with many-shot in-context examples.

Usage:
    python generate_game_data.py --game rps --num_games 10000 --output rps_games.npy
    python generate_game_data.py --game tictactoe --num_games 10000
    python generate_game_data.py --game poker --num_games 10000
    python generate_game_data.py --all --num_games 10000
"""

import argparse
import json
from pathlib import Path

import numpy as np


# =============================================================================
# Rock-Paper-Scissors
# =============================================================================
class RPSGenerator:
    """
    Generate Rock-Paper-Scissors game sequences.

    Tokens: R=0, P=1, S=2

    Each "game" is a sequence of (player1_move, player2_move) pairs,
    encoded as a flat sequence of tokens.

    Strategies for player 2:
    - "random": Uniform random
    - "counter": Tries to counter player 1's most recent move
    - "pattern": Follows a repeating pattern
    - "win_stay_lose_shift": Win-Stay Lose-Shift strategy
    - "mixed": Mix of strategies (more realistic)
    """

    name = "rps"
    vocab_size = 3
    token_names = ["R", "P", "S"]

    BEATS = {0: 2, 1: 0, 2: 1}  # R beats S, P beats R, S beats P
    COUNTER = {0: 1, 1: 2, 2: 0}  # Counter: P beats R, S beats P, R beats S

    @staticmethod
    def generate_sequence(length, strategy="mixed", rng=None):
        """Generate a sequence of RPS plays.

        Args:
            length: Number of rounds (output sequence length = 2 * length)
            strategy: Strategy for player 2
            rng: Random number generator

        Returns:
            sequence: Array of tokens [p1_move, p2_move, p1_move, p2_move, ...]
            metadata: Dict with game info
        """
        if rng is None:
            rng = np.random.default_rng()

        seq = []
        p2_last = rng.integers(0, 3)
        p1_last = rng.integers(0, 3)

        for t in range(length):
            # Player 1: random (represents the "environment")
            p1 = rng.integers(0, 3)

            # Player 2: depends on strategy
            if strategy == "random":
                p2 = rng.integers(0, 3)
            elif strategy == "counter":
                # Counter player 1's last move
                p2 = RPSGenerator.COUNTER[p1_last] if t > 0 else rng.integers(0, 3)
            elif strategy == "pattern":
                p2 = t % 3
            elif strategy == "win_stay_lose_shift":
                if t == 0:
                    p2 = rng.integers(0, 3)
                else:
                    # Check if p2 won last round
                    if RPSGenerator.BEATS[p2_last] == p1_last:
                        p2 = p2_last  # Stay
                    else:
                        p2 = rng.integers(0, 3)  # Shift
            elif strategy == "mixed":
                # Choose a strategy randomly each round
                s = rng.choice(["random", "counter", "pattern"])
                if s == "random":
                    p2 = rng.integers(0, 3)
                elif s == "counter":
                    p2 = RPSGenerator.COUNTER[p1_last] if t > 0 else rng.integers(0, 3)
                else:
                    p2 = t % 3
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            seq.extend([p1, p2])
            p1_last, p2_last = p1, p2

        return np.array(seq, dtype=np.int32)

    @staticmethod
    def generate_dataset(num_games, rounds_per_game=50, strategy="mixed", seed=42):
        rng = np.random.default_rng(seed)
        seq_len = rounds_per_game * 2
        data = np.zeros((num_games, seq_len), dtype=np.int32)
        for i in range(num_games):
            data[i] = RPSGenerator.generate_sequence(rounds_per_game, strategy, rng)
        return data


# =============================================================================
# Tic-Tac-Toe
# =============================================================================
class TicTacToeGenerator:
    """
    Generate Tic-Tac-Toe game transcripts.

    Board positions 0-8 (row-major):
      0 | 1 | 2
      ---------
      3 | 4 | 5
      ---------
      6 | 7 | 8

    Tokens: positions 0-8 represent moves, token 9 = game separator
    Each game is a sequence of moves, alternating X and O.

    Vocab size: 10 (9 positions + separator)
    """

    name = "tictactoe"
    vocab_size = 10  # 0-8 positions + 9 separator

    WIN_CONDITIONS = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6],           # diagonals
    ]

    @staticmethod
    def check_winner(board):
        """Check if there's a winner. Returns 1 (X), -1 (O), or 0."""
        for combo in TicTacToeGenerator.WIN_CONDITIONS:
            vals = [board[i] for i in combo]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return vals[0]
        return 0

    @staticmethod
    def generate_game(rng=None, strategy="random"):
        """Generate one tic-tac-toe game.

        Args:
            rng: Random number generator
            strategy: "random" for uniform random, "smart" for minimax-like play

        Returns:
            moves: List of move positions
            winner: 1 (X wins), -1 (O wins), 0 (draw)
        """
        if rng is None:
            rng = np.random.default_rng()

        board = [0] * 9  # 0=empty, 1=X, -1=O
        moves = []
        current = 1  # X goes first

        for turn in range(9):
            available = [i for i in range(9) if board[i] == 0]
            if not available:
                break

            if strategy == "smart" and rng.random() > 0.3:
                # Simple heuristic: try to win, then block, then center, then random
                move = TicTacToeGenerator._smart_move(board, current, available, rng)
            else:
                move = rng.choice(available)

            board[move] = current
            moves.append(move)

            winner = TicTacToeGenerator.check_winner(board)
            if winner != 0:
                return moves, winner

            current *= -1

        return moves, 0

    @staticmethod
    def _smart_move(board, player, available, rng):
        """Simple heuristic move selection."""
        # Try to win
        for m in available:
            board[m] = player
            if TicTacToeGenerator.check_winner(board) == player:
                board[m] = 0
                return m
            board[m] = 0

        # Try to block
        opp = -player
        for m in available:
            board[m] = opp
            if TicTacToeGenerator.check_winner(board) == opp:
                board[m] = 0
                return m
            board[m] = 0

        # Take center if available
        if 4 in available:
            return 4

        # Random
        return rng.choice(available)

    @staticmethod
    def generate_dataset(num_games, strategy="random", seed=42):
        """Generate dataset of tic-tac-toe games.

        Returns padded array where -1 = padding.
        """
        rng = np.random.default_rng(seed)
        games = []
        outcomes = []

        for _ in range(num_games):
            moves, winner = TicTacToeGenerator.generate_game(rng, strategy)
            games.append(moves)
            outcomes.append(winner)

        # Pad to max length (9)
        max_len = 9
        data = np.full((num_games, max_len), -1, dtype=np.int32)
        for i, game in enumerate(games):
            data[i, :len(game)] = game

        return data, np.array(outcomes, dtype=np.int32)


# =============================================================================
# Simple Poker
# =============================================================================
class SimplePokerGenerator:
    """
    Generate simplified poker hand sequences for studying hidden state inference.

    Simplified model (Kuhn Poker variant):
    - Deck: 3 cards (J=0, Q=1, K=2)
    - 2 players, each dealt 1 card
    - Actions: Check=0, Bet=1, Call=2, Fold=3
    - Higher card wins at showdown

    Sequence format per hand:
    [card_p1, card_p2, action1, action2, ..., result]

    Tokens: 0-2 = cards (J,Q,K), 3-6 = actions (check,bet,call,fold), 7 = p1_wins, 8 = p2_wins
    Vocab size: 9

    This is essentially Kuhn poker, well-studied in game theory.
    """

    name = "poker"
    vocab_size = 9
    token_names = ["J", "Q", "K", "check", "bet", "call", "fold", "p1_wins", "p2_wins"]

    # Token IDs
    CARD_J, CARD_Q, CARD_K = 0, 1, 2
    CHECK, BET, CALL, FOLD = 3, 4, 5, 6
    P1_WIN, P2_WIN = 7, 8

    @staticmethod
    def generate_hand(rng=None, p1_strategy="mixed", p2_strategy="mixed"):
        """Generate one hand of simplified poker.

        Returns sequence of tokens representing the hand.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Deal cards
        cards = rng.permutation(3)[:2]
        p1_card, p2_card = int(cards[0]), int(cards[1])

        seq = [p1_card, p2_card]

        # P1 acts first
        p1_action = SimplePokerGenerator._decide(p1_card, p1_strategy, rng, is_response=False)
        seq.append(p1_action)

        if p1_action == SimplePokerGenerator.CHECK:
            # P2 can check or bet
            p2_action = SimplePokerGenerator._decide(p2_card, p2_strategy, rng, is_response=False)
            seq.append(p2_action)

            if p2_action == SimplePokerGenerator.CHECK:
                # Showdown
                winner = SimplePokerGenerator.P1_WIN if p1_card > p2_card else SimplePokerGenerator.P2_WIN
                seq.append(winner)
            elif p2_action == SimplePokerGenerator.BET:
                # P1 must call or fold
                p1_response = SimplePokerGenerator._decide(p1_card, p1_strategy, rng, is_response=True)
                seq.append(p1_response)
                if p1_response == SimplePokerGenerator.CALL:
                    winner = SimplePokerGenerator.P1_WIN if p1_card > p2_card else SimplePokerGenerator.P2_WIN
                    seq.append(winner)
                else:  # Fold
                    seq.append(SimplePokerGenerator.P2_WIN)

        elif p1_action == SimplePokerGenerator.BET:
            # P2 must call or fold
            p2_response = SimplePokerGenerator._decide(p2_card, p2_strategy, rng, is_response=True)
            seq.append(p2_response)
            if p2_response == SimplePokerGenerator.CALL:
                winner = SimplePokerGenerator.P1_WIN if p1_card > p2_card else SimplePokerGenerator.P2_WIN
                seq.append(winner)
            else:  # Fold
                seq.append(SimplePokerGenerator.P1_WIN)

        return seq

    @staticmethod
    def _decide(card, strategy, rng, is_response=False):
        """Decide an action based on card and strategy."""
        if is_response:
            # Must call or fold
            if strategy == "random":
                return rng.choice([SimplePokerGenerator.CALL, SimplePokerGenerator.FOLD])
            elif strategy == "mixed":
                # Better cards more likely to call
                call_prob = 0.3 + 0.3 * card  # J:0.3, Q:0.6, K:0.9
                return SimplePokerGenerator.CALL if rng.random() < call_prob else SimplePokerGenerator.FOLD
            else:
                return SimplePokerGenerator.CALL if card >= 1 else SimplePokerGenerator.FOLD
        else:
            # Can check or bet
            if strategy == "random":
                return rng.choice([SimplePokerGenerator.CHECK, SimplePokerGenerator.BET])
            elif strategy == "mixed":
                bet_prob = 0.2 + 0.3 * card  # J:0.2, Q:0.5, K:0.8
                return SimplePokerGenerator.BET if rng.random() < bet_prob else SimplePokerGenerator.CHECK
            else:
                return SimplePokerGenerator.BET if card >= 1 else SimplePokerGenerator.CHECK

    @staticmethod
    def generate_dataset(num_hands, seed=42):
        """Generate dataset of poker hands.

        Returns padded array of hand sequences.
        """
        rng = np.random.default_rng(seed)
        hands = []

        for _ in range(num_hands):
            hand = SimplePokerGenerator.generate_hand(rng)
            hands.append(hand)

        max_len = max(len(h) for h in hands)
        data = np.full((num_hands, max_len), -1, dtype=np.int32)
        for i, hand in enumerate(hands):
            data[i, :len(hand)] = hand

        return data


# =============================================================================
# Main
# =============================================================================

GAMES = {
    "rps": RPSGenerator,
    "tictactoe": TicTacToeGenerator,
    "poker": SimplePokerGenerator,
}


def main():
    parser = argparse.ArgumentParser(description="Generate simple game datasets")
    parser.add_argument("--game", choices=list(GAMES.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--num_games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    games_to_run = list(GAMES.keys()) if args.all else [args.game]
    if not args.all and args.game is None:
        parser.error("Must specify --game or --all")

    for game_name in games_to_run:
        print(f"Generating {game_name}: {args.num_games} games...")

        if game_name == "rps":
            data = RPSGenerator.generate_dataset(args.num_games, seed=args.seed)
            np.save(output_dir / "rps_games.npy", data)
            print(f"  Saved: rps_games.npy (shape: {data.shape})")
            print(f"  Token distribution: {np.bincount(data.flatten(), minlength=3)}")

        elif game_name == "tictactoe":
            data, outcomes = TicTacToeGenerator.generate_dataset(args.num_games, seed=args.seed)
            np.save(output_dir / "tictactoe_games.npy", data)
            np.save(output_dir / "tictactoe_outcomes.npy", outcomes)
            x_wins = (outcomes == 1).sum()
            o_wins = (outcomes == -1).sum()
            draws = (outcomes == 0).sum()
            print(f"  Saved: tictactoe_games.npy (shape: {data.shape})")
            print(f"  Outcomes: X wins={x_wins}, O wins={o_wins}, draws={draws}")

        elif game_name == "poker":
            data = SimplePokerGenerator.generate_dataset(args.num_games, seed=args.seed)
            np.save(output_dir / "poker_hands.npy", data)
            print(f"  Saved: poker_hands.npy (shape: {data.shape})")


if __name__ == "__main__":
    main()
