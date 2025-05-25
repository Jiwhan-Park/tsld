# Approximate Thompson Sampling for Learning Linear Quadratic Regulators with $O(\sqrt{T})$ Regret

## 1. Requirements
- Python (>= 3.13)
- NumPy (>= 2.2.2)
- SciPy (>= 1.15.2)
- Matplotlib (>= 3.10.1)
- control (>= 0.10.1)
- PyTorch (>= 2.6.0)

## 2. Regret of system with non-Gaussian noise

### 2.1 Symmetric Gaussian mixture noise

In the case of symmetric Gaussian mixture noise, run the files: `gaussian_mixture_3D.py`, `gaussian_mixture_5D.py`, and `gaussian_mixture_10D.py`.

### 2.2 Asymmetric Gaussian mixture noise

In the case of asymmetric Gaussian mixture noise, run the files: `asymmetric_gaussian_mixture_3D.py`, `asymmetric_gaussian_mixture_5D.py`, and `asymmetric_gaussian_mixture_10D.py`.

## 3. Comparison

### 3.1 Comparison of Langevin iteration (naive vs preconditioned ULA)

To compare number of iterations for naive and preconditioned ULA, run `compare_Langevin_iteration.py`.

## 4. Plotting tools

You can plot various output files of our code using `plotfromcsv.py`.