"""
dataset.py
PyTorch Dataset and DataLoader utilities for Sudoku puzzles.
Uses Python's built-in csv module (zero extra dependencies required).
Compatible with standard Kaggle, HuggingFace, and custom CSV formats.
"""

import os
import csv
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from board_utils import to_one_hot


class SudokuDataset(Dataset):
    """
    Loads Sudoku puzzles and solutions from a CSV file.
    Expected CSV columns: 'quizzes'/'puzzle' and 'solutions'/'solution'.
    """
    def __init__(self, csv_path: str, max_samples: Optional[int] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")

        self.quizzes = []
        self.solutions = []

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Detect header names
            fieldnames = [fn.lower().strip() for fn in reader.fieldnames]
            quiz_col = next((fn for fn in reader.fieldnames if fn.lower().strip() in ['quizzes', 'puzzle', 'quiz']), reader.fieldnames[0])
            sol_col = next((fn for fn in reader.fieldnames if fn.lower().strip() in ['solutions', 'solution', 'sol']), reader.fieldnames[1] if len(reader.fieldnames) > 1 else reader.fieldnames[0])

            for idx, row in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break
                q = row[quiz_col].strip()
                s = row[sol_col].strip()
                if len(q) == 81 and len(s) == 81:
                    self.quizzes.append(q)
                    self.solutions.append(s)

        print(f"Loaded {len(self.quizzes):,} Sudoku puzzle pairs from {os.path.basename(csv_path)}.")

    def __len__(self) -> int:
        return len(self.quizzes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        quiz_str = self.quizzes[idx]
        sol_str = self.solutions[idx]

        quiz_board = np.array([int(c) for c in quiz_str], dtype=int).reshape(9, 9)
        sol_board = np.array([int(c) for c in sol_str], dtype=int).reshape(9, 9)

        # Convert input quiz to (9, 9, 9) one-hot float tensor
        quiz_tensor = torch.from_numpy(to_one_hot(quiz_board))

        # Target solution labels are classes 0..8 (representing digits 1..9) for CrossEntropyLoss
        sol_labels = torch.from_numpy(sol_board - 1).long()

        return quiz_tensor, sol_labels


def get_dataloaders(csv_path: str, batch_size: int = 64, split_ratio: float = 0.9, max_samples: Optional[int] = None):
    """Creates training and validation DataLoaders."""
    full_dataset = SudokuDataset(csv_path, max_samples=max_samples)
    
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader


if __name__ == "__main__":
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_sudoku.csv")
    dataset = SudokuDataset(sample_path)
    quiz, target = dataset[0]

    print("\n--- Dataset Sanity Check ---")
    print(f"Quiz one-hot shape  : {quiz.shape} (Channels=9, Rows=9, Cols=9)")
    print(f"Target labels shape : {target.shape} (Rows=9, Cols=9, values in 0..8)")
    print(f"Target sample (row 0): {target[0].numpy() + 1}")
