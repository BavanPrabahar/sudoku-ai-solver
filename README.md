# 🧠 AI-Powered Sudoku Solver

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red?style=flat-square&logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An elegant, high-performance Sudoku Solver combining **Deep Learning (Convolutional Neural Networks)** with an **Exact Backtracking Algorithm**. Built entirely in PyTorch, this repository demonstrates how machines can learn abstract spatial constraints, relational reasoning, and logic propagation without relying on hardcoded game rules.

Whether you're exploring neural networks, looking for a robust backtracking algorithm, or simply wanting to solve complex Sudoku grids instantly, this project provides both modular Python scripts and an interactive Jupyter Notebook environment.

---

## ✨ Features

- **Dual Solver Architecture**: Includes both a PyTorch-based Deep Learning model (SudokuCNN) and a precise Recursive Backtracking solver.
- **Categorical One-Hot Encoding**: Avoids numerical bias by representing each cell as a 9-channel binary vector.
- **Spatial 3x3 Convolutions**: Mirrors Sudoku's inherent subgrid constraints directly within the neural architecture.
- **Deep Receptive Field**: Utilizes a 9-layer stack to ensure every cell can cross-reference the entire 81-cell board.
- **High-Speed Inference**: Evaluates all 81 cells simultaneously, yielding solutions in milliseconds.

---

## 🗂️ Project Structure

```text
sudoku-ai-solver/
├── 📓 sudoku_ai_solver.ipynb      # Interactive Jupyter Notebook with walkthroughs
├── 📁 data/
│   └── sample_sudoku.csv          # Included starter dataset (puzzles & solutions)
├── ⚙️ board_utils.py              # Utilities for encoding, decoding, and validation
├── 🔍 backtracking_solver.py      # Ground-truth recursive backtracking implementation
├── 🧠 model.py                    # PyTorch CNN Architecture (9 layers, 64 hidden dimensions)
├── 📊 dataset.py                  # PyTorch DataLoader and Dataset for CSV ingestion
├── 🏋️ train.py                    # Training loop with accuracy and loss tracking
├── 🚀 solve.py                    # Inference script for benchmarking both solvers
├── 👁️ demo_activation_solve.py    # Walkthrough of single-cell activation competition
├── 📝 requirements.txt            # Project dependencies
└── 📖 README.md                   # You are here
```

---

## 🚀 Quickstart

### Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/BavanPrabahar/sudoku-ai-solver.git
cd sudoku-ai-solver
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage & Workflows

### 🔬 Option 1: Jupyter Notebook Experience
The easiest way to explore the project is via the included Jupyter Notebook:
Open `sudoku_ai_solver.ipynb` in your favorite IDE (VS Code, JupyterLab) or Google Colab to interact with the code cell-by-cell.

### ⚡ Option 2: Command-Line Interface

#### Solve a Custom Puzzle
Pass a string of 81 characters (use `0` for empty cells) to solve it using both the Neural Network and Backtracking algorithms:
```bash
python solve.py --puzzle "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
```

#### Train the Deep Learning Model
Train the CNN from scratch using the included starter data. You can also supply a larger dataset via the `--csv_path` argument:
```bash
python train.py --csv_path data/sample_sudoku.csv --epochs 10 --batch_size 64 --lr 0.001
```

#### Run the Exact Backtracking Benchmark
Test the raw speed and accuracy of the recursive algorithmic approach:
```bash
python backtracking_solver.py
```

#### Visualize Neural Activations
Inspect how the model makes decisions on a per-cell basis through activation scoring:
```bash
python demo_activation_solve.py
```

---

## 📊 Datasets

For large-scale training, we recommend the following verified Kaggle datasets. A small verified sample is provided in `data/sample_sudoku.csv` to get you started immediately.

| Dataset | Size | Format | Direct Link |
| :--- | :--- | :--- | :--- |
| **Kaggle 1M Sudoku Games** | 1,000,000 puzzles | CSV (`quizzes,solutions`) | [View on Kaggle](https://www.kaggle.com/datasets/bryanpark/sudoku) |
| **Kaggle 3M Rated Puzzles** | 3,000,000 puzzles | CSV (with difficulty ratings) | [View on Kaggle](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings) |

---

## 📄 License
This project is licensed under the **MIT License**. It is free for educational, research, and commercial use. See the source code for more details.

---
*Maintained by [BavanPrabahar](https://github.com/BavanPrabahar)*
