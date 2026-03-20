#!/usr/bin/env python3
"""
Generate synthetic Othello game transcripts for OthelloGPT-style experiments.

This uses the OthelloBoardState class from the othello_world repository
(Li et al., 2023) to generate sequences of legal Othello moves by random play.

Each game is a sequence of integers in [0, 59] representing moves on the 8x8
board (excluding the 4 center squares which are pre-filled).

Usage:
    python generate_othello_games.py --num_games 10000 --output othello_games.npy
    python generate_othello_games.py --num_games 100 --output sample.npy --verbose

References:
    - Li et al. "Emergent World Representations: Exploring a Sequence Model
      Trained on a Synthetic Task" (ICLR 2023)
    - https://github.com/likenneth/othello_world
"""

import argparse
import sys
import os
import random
import multiprocessing
from pathlib import Path

import numpy as np

# Standalone OthelloBoardState adapted from othello_world/data/othello.py
# (Li et al., 2023). Extracted here to avoid heavy dependencies (pgn,
# matplotlib, seaborn, psutil) in the original module.

_eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

class OthelloBoardState:
    """Othello board logic. 1=black, -1=white."""
    def __init__(self, board_size=8):
        self.board_size = board_size * board_size
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.state = board.copy()
        self.next_hand_color = 1
        self.history = []

    def get_state(self):
        return (self.state + 1).flatten().tolist()  # 0=white, 1=empty, 2=black

    def update(self, moves):
        for move in moves:
            self._umpire(move)

    def _umpire(self, move):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        color = self.next_hand_color
        tbf = self._find_flips(r, c, color)
        if len(tbf) == 0:
            color *= -1
            self.next_hand_color *= -1
            tbf = self._find_flips(r, c, color)
        assert len(tbf) > 0, "Illegal move!"
        for fr, fc in tbf:
            self.state[fr, fc] *= -1
        self.state[r, c] = color
        self.next_hand_color *= -1
        self.history.append(move)

    def _find_flips(self, r, c, color):
        tbf = []
        for dr, dc in _eights:
            buffer = []
            cr, cc = r, c
            while True:
                cr, cc = cr + dr, cc + dc
                if cr < 0 or cr > 7 or cc < 0 or cc > 7:
                    break
                if self.state[cr, cc] == 0:
                    break
                elif self.state[cr, cc] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cr, cc])
        return tbf

    def _tentative_move(self, move):
        r, c = move // 8, move % 8
        if self.state[r, c] != 0:
            return 0
        color = self.next_hand_color
        if self._find_flips(r, c, color):
            return 1
        if self._find_flips(r, c, -color):
            return 2
        return 0

    def get_valid_moves(self):
        regular, forfeit = [], []
        for move in range(64):
            x = self._tentative_move(move)
            if x == 1:
                regular.append(move)
            elif x == 2:
                forfeit.append(move)
        return regular or forfeit or []


def generate_single_game(_seed=None):
    """Generate a single random legal Othello game.

    Returns:
        list[int]: Sequence of moves (board positions 0-63).
    """
    moves = []
    board = OthelloBoardState()
    valid = board.get_valid_moves()
    while valid:
        move = random.choice(valid)
        moves.append(move)
        board.update([move])
        valid = board.get_valid_moves()
    return moves


def generate_single_game_with_boards(_seed=None):
    """Generate a game along with board states at each step.

    Returns:
        tuple: (moves: list[int], boards: list[list[int]])
            boards[i] is the board state AFTER move i.
            Board state values: 0=white, 1=empty, 2=black
    """
    moves = []
    boards = []
    board = OthelloBoardState()
    valid = board.get_valid_moves()
    while valid:
        move = random.choice(valid)
        moves.append(move)
        board.update([move])
        boards.append(board.get_state())
        valid = board.get_valid_moves()
    return moves, boards


# Mapping from 64-position space to 60-position space (excluding center 4)
CENTER_SQUARES = [27, 28, 35, 36]

def to_60_token(move_64):
    """Convert 0-63 move to 0-59 token (skipping center squares)."""
    offset = sum(1 for c in CENTER_SQUARES if move_64 > c)
    return move_64 - offset

def from_60_token(token_60):
    """Convert 0-59 token back to 0-63 move."""
    move = token_60
    for c in CENTER_SQUARES:
        if move >= c:
            move += 1
    return move


def generate_games(num_games, num_workers=None, use_60_tokens=True):
    """Generate multiple Othello games.

    Args:
        num_games: Number of games to generate.
        num_workers: Number of parallel workers (default: cpu_count).
        use_60_tokens: If True, map to 0-59 token space (default for OthelloGPT).

    Returns:
        list[list[int]]: List of game sequences.
    """
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    if num_workers > 1 and num_games > 100:
        with multiprocessing.Pool(num_workers) as pool:
            games = list(pool.imap(generate_single_game, range(num_games)))
    else:
        games = [generate_single_game() for _ in range(num_games)]

    if use_60_tokens:
        games = [[to_60_token(m) for m in game] for game in games]

    return games


def main():
    parser = argparse.ArgumentParser(description="Generate Othello game transcripts")
    parser.add_argument("--num_games", type=int, default=10000,
                       help="Number of games to generate (default: 10000)")
    parser.add_argument("--output", type=str, default="othello_games.npy",
                       help="Output filename (default: othello_games.npy)")
    parser.add_argument("--format", choices=["npy", "txt"], default="npy",
                       help="Output format (default: npy)")
    parser.add_argument("--token_space", type=int, choices=[60, 64], default=60,
                       help="Token space: 60 (skip center) or 64 (default: 60)")
    parser.add_argument("--with_boards", action="store_true",
                       help="Also save board states")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.num_games} Othello games...")
    use_60 = (args.token_space == 60)
    games = generate_games(args.num_games, num_workers=args.workers, use_60_tokens=use_60)

    # Pad to uniform length for numpy storage
    lengths = [len(g) for g in games]
    max_len = max(lengths)
    mean_len = np.mean(lengths)

    if args.verbose:
        print(f"  Game lengths: mean={mean_len:.1f}, min={min(lengths)}, max={max_len}")
        print(f"  Token space: {args.token_space}")

    output_path = Path(__file__).parent / args.output

    if args.format == "npy":
        # Pad with -1 for variable length
        padded = np.full((len(games), max_len), -1, dtype=np.int16)
        for i, game in enumerate(games):
            padded[i, :len(game)] = game
        np.save(output_path, padded)
        print(f"Saved {len(games)} games to {output_path} (shape: {padded.shape})")
    else:
        with open(output_path, "w") as f:
            for game in games:
                f.write(" ".join(str(m) for m in game) + "\n")
        print(f"Saved {len(games)} games to {output_path}")


if __name__ == "__main__":
    main()
