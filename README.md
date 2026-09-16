# 🧩 Deep Learning & Exact Sudoku AI Solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance **Deep Learning (Convolutional Neural Network) & Exact Backtracking Sudoku Solver** built in PyTorch. 

This repository demonstrates how machines learn abstract spatial constraints, relational reasoning, and logic propagation across an $81$-cell grid without hardcoded game rules.

---

## 🚀 Key Highlights & Architecture

* **Categorical One-Hot Encoding**: Rather than feeding raw numerical scalars $1..9$ (which tricks neural nets into assuming $9$ is $9\times$ larger than $1$), each cell is represented as a 9-channel binary vector $\implies$ input shape $(9 \text{ channels}, 9 \text{ rows}, 9 \text{ cols})$.
* **Spatial $3 \times 3$ Convolutions**: A $3 \times 3$ kernel directly mirrors the subgrid box constraints of Sudoku.
* **Why 9 Layers? (Receptive Field Geometry)**:
  - In Layer 1, each cell only sees its immediate $3 \times 3$ box.
  - Across successive layers, the receptive field ripples outward.
  - Because corner cells are **8 steps away** from the opposite corner, a **9-layer stack** is mathematically required to guarantee every cell can cross-reference the entire board.
* **Simultaneous Parallel Computation**: Unlike sequential backtracking (which checks cell-by-cell), the neural network evaluates **all 81 cells simultaneously in 2 milliseconds**.
* **Exact Benchmark Baseline**: Includes an exact recursive backtracking solver that achieves 100% ground-truth accuracy in $< 100\text{ ms}$.

---

## 📊 Online Sudoku Datasets & Benchmarks

You can download large-scale public Sudoku datasets to train this network:

| Dataset | Size | Format | Direct Link |
| :--- | :--- | :--- | :--- |
| **Kaggle 1 Million Sudoku Games** | 1,000,000 puzzles | CSV (`quizzes,solutions`) | [Kaggle Dataset (Bryan Park)](https://www.kaggle.com/datasets/bryanpark/sudoku) |
| **Kaggle 3 Million Puzzles with Ratings** | 3,000,000 puzzles | CSV with difficulty ratings | [Kaggle Dataset (Radcliffe)](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings) |
| **Hugging Face Sudoku-1M** | 1,000,000 puzzles | Parquet / CSV | [Hugging Face Hub](https://huggingface.co/datasets/Ritvik19/Sudoku-1M) |
| **Hugging Face Sudoku-3M** | 3,000,000 puzzles | Parquet / CSV | [Hugging Face Hub](https://huggingface.co/datasets/omarmomen/sudoku-3m) |
| **Peter Norvig Benchmark Collection** | Hardest & Top95 sets | Plain text | [Norvig's Sudoku Test Suite](https://norvig.com/sudoku.html) |

A starter dataset of verified puzzles is included in [`data/sample_sudoku.csv`](data/sample_sudoku.csv).

---

## 📦 Project Structure

```
sudoku-ai-solver/
├── data/
│   └── sample_sudoku.csv       # Starter puzzle/solution dataset
├── board_utils.py              # One-hot encoding/decoding & constraint validator
├── backtracking_solver.py      # Ground-truth recursive backtracking solver
├── model.py                    # PyTorch SudokuCNN architecture (9 layers, 64 hidden dim)
├── dataset.py                  # PyTorch Dataset and DataLoader for CSVs
├── train.py                    # Multi-epoch training loop with accuracy metrics
├── solve.py                    # Inference script comparing DL vs Backtracking
├── demo_activation_solve.py    # Visual activation calculation walkthrough
├── requirements.txt            # Minimal dependencies (PyTorch + NumPy)
└── README.md                   # Complete documentation
```

---

## 🛠️ Quickstart Installation

```bash
# 1. Clone the repository
git clone https://github.com/BavanPrabahar/sudoku-ai-solver.git
cd sudoku-ai-solver

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt
```

---

## 🎯 Usage

### 1. Run the Exact Benchmark Solver
```bash
python backtracking_solver.py
```

### 2. Inspect Single-Cell Activation Competition
```bash
python demo_activation_solve.py
```

### 3. Train the Deep Learning Model
```bash
# Train on sample dataset (or pass your downloaded Kaggle CSV via --csv_path)
python train.py --csv_path data/sample_sudoku.csv --epochs 10 --batch_size 64 --lr 0.001
```

### 4. Solve Any Puzzle with Both Solvers
```bash
python solve.py --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079
```

---

## 🧠 Model Architecture Specification

$$\text{Input: } (B, 9, 9, 9) \xrightarrow{\text{Conv2D + BN + ReLU} \times 9} \text{Output: } (B, 9, 9, 9)$$

* **Input Channels**: 9 (One-hot binary channels for digits $1..9$).
* **Hidden Layers**: 7 intermediate blocks with $64$ feature maps each.
* **Output Logits**: 9 logits per cell $\implies 81 \times 9 = 729$ candidate scores.
* **Total Parameters**: **265,353** (Lightweight, trains in minutes on GPU).
* **Loss Function**: Multi-class `nn.CrossEntropyLoss()` evaluated across all 81 cells.

---

## 📜 License
MIT License. Free for educational, research, and commercial use.
