# ML on Predicting Chaotic Systems

Predict short-horizon chaotic system dynamics from time series using:

- **LSTM (PyTorch)**
- **Reservoir Computing (RC / ESN) with [ReservoirPy](https://github.com/reservoirpy/reservoirpy)**
- **Next-Generation Reservoir Computing (NG-RC) via ReservoirPy `NVAR` + ridge regression**

The project includes experiments on common chaotic attractors (e.g., **Lorenz**, **Rössler**, **Chen**, **Qi**) and a noisy-data setting.

![Chaotic systems](code/Figure/chaotic%20systems.png)

## What’s in this repo

- **Datasets**: multivariate time series in `code/Dataset/*.csv`
  - Column 0: time
  - Columns 1–3: system states (x, y, z)
- **Experiments (Jupyter notebooks)**: `code/Experiments/`
  - `LSTM.ipynb`: PyTorch LSTM forecaster (uses `reservoirpy.observables` for `nrmse`/`rsquare`)
  - `RC.ipynb`: ESN / reservoir computing with ReservoirPy (`Reservoir >> Ridge`)
  - `NGRC.ipynb`: NG-RC with ReservoirPy (`NVAR >> Ridge`)
  - `noise_*.ipynb`: noisy-data experiments (e.g., noisy Lorenz)
- **Figures**: `code/Figure/Task1/`, `code/Figure/Task2/`

## Methods (high level)

- **One-step forecasting**: learn \( \hat{x}_{t+1} = f(x_t) \) on \((x_t, x_{t+1})\) pairs (3D states).
- **LSTM**: sequence model trained with MSE loss on normalized data; rolled forward for prediction.
- **RC / ESN (ReservoirPy)**: fixed random recurrent reservoir + trained linear readout (ridge regression).
- **NG-RC (ReservoirPy NVAR)**: feature expansion of delayed coordinates (nonlinear vector autoregression) + ridge readout.
- **Metrics**: `NRMSE` and `R^2` (via `reservoirpy.observables`).

## Setup

### Prerequisites

- **Python 3.9+**
- Jupyter environment (JupyterLab or classic Notebook)

### Install dependencies

Create/activate an environment (recommended), then install:

```bash
pip install -U pip
pip install -U numpy pandas matplotlib jupyter reservoirpy torch
```

Notes:
- If you want GPU acceleration for the LSTM, install a CUDA-enabled PyTorch build following the official PyTorch instructions for your system.

## How to run

From the repo root:

```bash
jupyter lab
```

Then open and run notebooks in `code/Experiments/` top-to-bottom:

- **Task 1 (multiple systems)**:
  - `code/Experiments/LSTM.ipynb`
  - `code/Experiments/RC.ipynb`
  - `code/Experiments/NGRC.ipynb`
- **Task 2 (noisy setting)**:
  - `code/Experiments/noise_LSTM.ipynb`
  - `code/Experiments/noise_RC.ipynb`
  - `code/Experiments/noise_NGRC.ipynb`

## Repository structure

```text
.
├── code/
│   ├── Dataset/                 # CSV datasets for each chaotic system
│   ├── Experiments/             # Notebooks (LSTM, RC/ESN, NG-RC, noise variants)
│   ├── Figure/
│   │   ├── Task1/               # Clean-data plots and long-horizon visuals
│   │   └── Task2/               # Noisy-data plots
│   └── Generate_dataset/        # Dataset generation notebooks
└── README.md
```

## Results

Generated plots are saved under `code/Figure/` (Task-specific subfolders). Typical outputs include:

- Ground-truth trajectories vs model predictions (x/y/z)
- 3D attractor reconstructions
- Long-horizon rollout behavior comparisons

## Reproducibility tips

- If you re-run reservoirs, set a fixed seed (ReservoirPy supports `rpy.set_seed(...)`).
- Results can vary with hyperparameters (reservoir size, spectral radius, leak rate, ridge strength; or LSTM hidden size/layers/learning rate).

## Acknowledgements

- Reservoir computing tooling is provided by **ReservoirPy**.
