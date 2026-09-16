# 🧩 Deep Learning & Exact Sudoku AI Solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter%20%2F%20Colab-orange.svg)](sudoku_ai_solver.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance **Deep Learning (Convolutional Neural Network) & Exact Backtracking Sudoku Solver** built in PyTorch.

This repository demonstrates how machines learn abstract spatial constraints, relational reasoning, and logic propagation across an $81$-cell grid without hardcoded game rules.

Available as both modular Python scripts and an interactive, all-in-one Jupyter Notebook: [`sudoku_ai_solver.ipynb`](sudoku_ai_solver.ipynb).

---

## 📊 Verified Online Datasets

These two datasets are standard, verified, and widely used benchmarks on Kaggle:

| Dataset | Size | Format | Direct Verified Link |
| :--- | :--- | :--- | :--- |
| **Kaggle 1 Million Sudoku Games** *(Bryan Park)* | 1,000,000 puzzles | CSV (`quizzes,solutions`) | [Kaggle Dataset (Bryan Park)](https://www.kaggle.com/datasets/bryanpark/sudoku) |
| **Kaggle 3 Million Puzzles with Ratings** *(Grant Radcliffe)* | 3,000,000 puzzles | CSV with difficulty ratings | [Kaggle Dataset (Radcliffe)](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings) |

> [!NOTE]
> A starter dataset of verified puzzles is included out-of-the-box in [`data/sample_sudoku.csv`](data/sample_sudoku.csv) so you can train and test immediately without downloading large external files.

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

## 📦 Project Structure

```
sudoku-ai-solver/
├── sudoku_ai_solver.ipynb      # Complete all-in-one interactive Jupyter Notebook
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

### Option A: Run the Interactive Jupyter Notebook
Open [`sudoku_ai_solver.ipynb`](sudoku_ai_solver.ipynb) in Jupyter, VS Code, or Google Colab and run through the cells!

### Option B: Run via Command Line

#### 1. Run the Exact Benchmark Solver
```bash
python backtracking_solver.py
```

#### 2. Inspect Single-Cell Activation Competition
```bash
python demo_activation_solve.py
```

#### 3. Train the Deep Learning Model
```bash
# Train on sample dataset (or pass your downloaded Kaggle CSV via --csv_path)
python train.py --csv_path data/sample_sudoku.csv --epochs 10 --batch_size 64 --lr 0.001
```

#### 4. Solve Any Puzzle with Both Solvers
```bash
python solve.py --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079
```

---

## 📜 License
MIT License. Free for educational, research, and commercial use.
