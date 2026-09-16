"""
model.py
Deep Learning Sudoku Solver using a Convolutional Neural Network (CNN) in PyTorch.
"""

import torch
import torch.nn as nn


class SudokuCNN(nn.Module):
    """
    A Deep Convolutional Neural Network designed for Sudoku reasoning.
    
    Architecture:
    - Input: (Batch, 9 channels, 9 rows, 9 cols)
    - 1st Conv Layer: Projects 9 channels to 64 feature maps
    - Intermediate Conv Blocks: Stacked Conv2d (3x3) + BatchNorm + ReLU
      (Allows each cell to share information across its row, column, and 3x3 box)
    - Output Conv Layer: Projects back to 9 channels (logits for digits 1..9)
    """
    def __init__(self, num_layers: int = 9, hidden_dim: int = 64):
        super().__init__()
        
        layers = []
        # Input layer: 9 input channels -> hidden_dim
        layers.append(nn.Conv2d(in_channels=9, out_channels=hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU())
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU())
            
        # Output layer: hidden_dim -> 9 output channels (logits for each digit)
        layers.append(nn.Conv2d(in_channels=hidden_dim, out_channels=9, kernel_size=1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 9, 9, 9) representing one-hot boards.
        Returns:
            logits: Tensor of shape (Batch, 9, 9, 9) representing digit predictions.
        """
        return self.network(x)


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SudokuCNN(num_layers=9, hidden_dim=64).to(device)
    
    # Simulate a batch of 4 empty/sample Sudoku boards
    dummy_input = torch.randn(4, 9, 9, 9, device=device)
    output = model(dummy_input)
    
    print("\n--- Model Sanity Check ---")
    print(f"Input shape  : {dummy_input.shape} (Batch=4, Channels=9, Rows=9, Cols=9)")
    print(f"Output shape : {output.shape} (Batch=4, Channels=9, Rows=9, Cols=9)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
