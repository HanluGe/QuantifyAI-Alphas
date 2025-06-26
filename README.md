# QuantifyAI-Alphas: Alpha Factor Generation and Evaluation Framework

QuantifyAI-Alphas is a research-oriented framework for generating, evaluating, and ranking quantitative alpha factors. It supports futures, equities, and other asset classes and integrates reinforcement learning, QLib data, expression trees, and benchmark evaluations.

## Project Structure

```python
alphagen-futures/
├── alphagen/             # Core modules: models, strategies, data structures
│   ├── trade/            # Trading logic
│   ├── rl/               # Reinforcement learning policies
│   ├── models/           # Alpha pool management
│   ├── data/             # Expression trees and tokens
│   ├── utils/            # Utility functions (correlation, etc.)
│   └── config.py         # Configuration
│
├── alphagen_generic/     # Generic operators and feature functions
│   ├── features.py
│   └── operators.py
│
├── alphagen_qlib/        # QLib integration and data handling
│   ├── calculator.py     # IC, IR, and backtest metrics
│   ├── utils.py
│   ├── stock_data.py
│   └── strategy.py
│
├── data/                 # Local raw data directory
│
├── alpha_ranking.py      # Main workflow for alpha ranking and export
├── main.py               # Entry point for alpha training and generation
├── requirements.txt      # Python dependencies
└── Untitled.ipynb        # Experimental notebook
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Train and evaluate alpha factors:

```bash
python main.py
```

Rank and export top-performing alphas:

```bash
python alpha_ranking.py
```

## Features

- Expression-tree-based alpha generation
- Multi-market support (futures, equities, crypto)
- Integrated reinforcement learning policy training
- Benchmark-based alpha ranking and decorrelation
- QLib integration and backtesting support
