# ML on Predicting Chaotic Systems
Predict chaotic system dynamics using:

- **Long short-term memory (LSTM) with [PyTorch](https://pytorch.org/)**
- **Reservoir Computing (RC) with [ReservoirPy](https://github.com/reservoirpy/reservoirpy)**
- **Next-Generation Reservoir Computing (NG-RC) with [ReservoirPy](https://github.com/reservoirpy/reservoirpy)**

The project includes experiments on common 3-D chaotic attractors (e.g., **Lorenz**, **Rössler**, **Chen**, **Qi**) and a noisy-data setting.

![Chaotic systems](Figure/chaotic%20systems.png)

## What’s in this repo

- **Datasets**: multivariate time series in `Dataset/*.csv`
  - Column 0: time
  - Columns 1–3: system states (x, y, z)
- **Experiments (Jupyter notebooks)**: `code/Experiments/`
  - `LSTM.ipynb`: PyTorch LSTM model 
  - `RC.ipynb`: RC with ReservoirPy (`Reservoir >> Ridge`)
  - `NGRC.ipynb`: NG-RC with ReservoirPy (`NVAR >> Ridge`)
  - `noise_*.ipynb`: noisy-data experiments (e.g., noisy Lorenz)
- **Figures**: `Figure/Task1/`, `Figure/Task2/`

## Methods (high level)

- **Multi-step forecasting (autoregressive rollout)**: train a one-step model $\hat{x}(t+1) = f(x(t))$, then generate multi-step predictions by feeding $\hat{x}(t+1)$ back as input to predict $\hat{x}(t+2), \hat{x}(t+3), \dots$  
- **LSTM**: sequence model trained with MSE loss on normalized data; rolled forward autoregressively for multi-step prediction.
- **RC**: fixed random recurrent reservoir + trained linear readout (ridge regression).
- **NG-RC**: feature expansion of delayed coordinates (nonlinear vector autoregression) + ridge readout.
- **Metrics**: `NRMSE` and `R^2` (via `reservoirpy.observables`).

## Setup

### Prerequisites

- Python 3.9+
- Jupyter environment (JupyterLab or classic Notebook)

### Install dependencies

Create/activate an environment (recommended), then install:

```bash
pip install -U numpy pandas matplotlib jupyter reservoirpy torch
```

Notes:
- If you want GPU acceleration for the LSTM, install a CUDA-enabled PyTorch build following the official PyTorch instructions for your system.

## How to run

From the repo root:

```bash
jupyter lab
```

First run `code/Generate_dataset/generate_chaotic_dataset.ipynb` to generate all chaotic systems' time series. Feel free to change the time horizon variable `tspan`.

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
├── Dataset/                     # CSV datasets for each chaotic system
├── Figure/                      # Saved plots (Task-specific subfolders)
├── code/
│   ├── Experiments/             # Notebooks (LSTM, RC/ESN, NG-RC, noise variants)
│   └── Generate_dataset/        # Dataset generation notebooks
└── README.md
```

## Results

Generated plots are saved under `Figure/` (Task-specific subfolders). Typical outputs include:

- Ground-truth trajectories vs model predictions (x/y/z)
- 3D attractor reconstructions
- Long-horizon rollout behavior comparisons

## Cite this work
```
@article{NSCE05500,
  title={Predicting chaotic system behavior using machine learning techniques},
  author={Rao, Huaiyuan and Zhao, Yichen and Chen, Hsuan-Pin},
  journal={Nonlinear Science and Control Engineering},
  issn={TBA},
  volume={1},
  number={1},
  pages={025290003},
  doi={https://doi.org/10.36922/NSCE025290003},
  url={https://accscience.com/journal/NSCE/1/1/10.36922/NSCE025290003},
  year={2025}
}
```
