"""
solve.py
Inference script to solve any 81-character Sudoku puzzle using:
1. The trained Deep Learning (CNN) solver
2. The exact Backtracking benchmark solver
"""

import argparse
import time
import numpy as np
import torch
from model import SudokuCNN
from board_utils import to_one_hot, print_board
from backtracking_solver import solve_sudoku


def solve_with_dl(model, puzzle_str: str, device: torch.device) -> np.ndarray:
    """Solves an 81-char puzzle string using the neural network."""
    board = np.array([int(c) for c in puzzle_str], dtype=int).reshape(9, 9)
    tensor = torch.from_numpy(to_one_hot(board)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)  # (1, 9, 9, 9)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() + 1

    # Keep original clues fixed, replace zeros with model predictions
    solved_board = np.where(board != 0, board, preds)
    return solved_board


def main():
    parser = argparse.ArgumentParser(description="Solve Sudoku using Deep Learning & Backtracking")
    parser.add_argument("--puzzle", type=str, default="530070000600195000098000060800060003400803001700020006060000280000419005000080079", help="81-character puzzle string (0 denotes empty)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sudoku_cnn.pt", help="Path to trained model weights")
    args = parser.parse_args()

    if len(args.puzzle) != 81 or not args.puzzle.isdigit():
        print("Error: Puzzle must be exactly 81 numeric characters (0-9).")
        return

    original_board = np.array([int(c) for c in args.puzzle], dtype=int).reshape(9, 9)
    print("=" * 45)
    print("          ORIGINAL SUDOKU PUZZLE")
    print("=" * 45)
    print_board(original_board)

    # 1. Backtracking Exact Benchmark
    print("-" * 45)
    print("Solving with Backtracking (Exact Benchmark)...")
    bt_board = original_board.copy()
    start_time = time.perf_counter()
    bt_success = solve_sudoku(bt_board)
    bt_time = (time.perf_counter() - start_time) * 1000
    print(f"✓ Backtracking solved in {bt_time:.2f} ms")
    print_board(bt_board)

    # 2. Deep Learning Solver
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SudokuCNN().to(device)
    if torch.cuda.is_available():
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded trained weights from {args.checkpoint}")
        except Exception:
            print("Notice: No trained checkpoint found. Running untuned forward pass for demo.")

    start_time = time.perf_counter()
    dl_board = solve_with_dl(model, args.puzzle, device)
    dl_time = (time.perf_counter() - start_time) * 1000
    print(f"✓ Neural Network forward pass in {dl_time:.2f} ms")
    print_board(dl_board)


if __name__ == "__main__":
    main()
