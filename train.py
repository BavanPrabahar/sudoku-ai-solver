"""
train.py
Training loop for the Deep Learning SudokuCNN solver.
Tracks loss, cell-level accuracy, and full-board solution accuracy.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import SudokuCNN
from dataset import get_dataloaders


def evaluate(model, val_loader, criterion, device):
    """Evaluates model on validation dataset."""
    model.eval()
    total_loss = 0.0
    total_cells = 0
    correct_cells = 0
    total_boards = 0
    correct_boards = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)  # (Batch, 9, 9, 9)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            # Predictions: argmax across channel axis (1) -> values in 0..8
            preds = torch.argmax(outputs, dim=1)

            # Cell-level accuracy
            correct_cells += (preds == targets).sum().item()
            total_cells += targets.numel()

            # Board-level accuracy (all 81 cells in the board must be correct)
            board_correct = (preds == targets).view(inputs.size(0), -1).all(dim=1).sum().item()
            correct_boards += board_correct
            total_boards += inputs.size(0)

    avg_loss = total_loss / max(1, total_boards)
    cell_acc = (correct_cells / max(1, total_cells)) * 100.0
    board_acc = (correct_boards / max(1, total_boards)) * 100.0
    return avg_loss, cell_acc, board_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        max_samples=args.max_samples
    )

    model = SudokuCNN(num_layers=args.num_layers, hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized SudokuCNN ({args.num_layers} layers, {args.hidden_dim} hidden dim, {total_params:,} parameters)")
    print("-" * 65)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss, cell_acc, board_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Cell Acc: {cell_acc:.1f}% | "
              f"Full Board Solved: {board_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  --> Saved new best checkpoint to {args.save_path}")

    print("-" * 65)
    print("✓ Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SudokuCNN Solver")
    parser.add_argument("--csv_path", type=str, default="data/sample_sudoku.csv", help="Path to Sudoku CSV file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Number of feature channels")
    parser.add_argument("--num_layers", type=int, default=9, help="Number of Conv layers")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/Validation split ratio")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to load from CSV")
    parser.add_argument("--save_path", type=str, default="checkpoints/sudoku_cnn.pt", help="Checkpoint save path")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA acceleration")
    args = parser.parse_args()

    train(args)
