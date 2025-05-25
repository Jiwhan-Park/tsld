import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from scipy import stats
from control import dare
from gaussian_ULA import *

torch.set_default_dtype(torch.float64)

# Set this parameter to False in order to conduct simulation for preconditioned ULA
naive = True

# System parameters
n = 3
m = 3
simul = 100
time_horizon = 2000

ULA_iter = np.zeros((simul, math.ceil((math.sqrt(8 * time_horizon + 5) - 3) / 2)))

# System matrices
A = torch.tensor([[0.3, 0.1, 0.2],
                  [0.1, 0.4, 0.0],
                  [0.0, 0.7, 0.6]])
B = torch.tensor([[0.5, 0.4, 0.5],
                  [0.6, 0.3, 0.0],
                  [0.3, 0.0, 0.2]])
Q = 2 * np.eye(n)
R = np.eye(m)
W = np.eye(n)

# Admissible conditions
theta_bound = 20
J_bound = 20000
rho = 0.99

# Initial prior
lam = 5
theta_mean_ = 0.5 * torch.ones(n + m, n)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for simulation in range(simul):
    if naive:
        print(f"Naive ULA Simulation {simulation + 1}")
    else:
        print(f"Preconditioned ULA Simulation {simulation + 1}")
    t = 1
    t_k = 0
    k = 1
    x = np.zeros((n, 1))
    data = torch.empty(0, 2 * n + m, 1)
    while True:
        print(f"Time step {t}")
        T = k + 1
        t_k = t
        while True:
            start = time.time()
            it, arbi = gaussian_ULA(lam, theta_mean_, data, n, m, t_k, device, naive)
            ULA_iter[simulation, k - 1] = it
            end = time.time()
            print(f"ULA total time: {end - start:.4e}")
            print(arbi)

            a = arbi[:n, :].T.numpy()
            b = arbi[n:, :].T.numpy()

            # Solve discrete algebraic Riccati equation
            S, _, Gain = dare(a, b, Q, R)

            # Check stability conditions
            cond1 = np.linalg.norm(arbi) < theta_bound
            cond2 = np.trace(W @ S) < J_bound
            eigvals = np.linalg.eigvals((a - b @ Gain).T @ (a - b @ Gain))
            cond3 = np.max(eigvals) < rho

            if cond1 and cond2 and cond3:
                break
            print('Rejection')

        # System interaction loop
        while t <= t_k + T - 1:
            if t == t_k + T - 1:
                u = -Gain @ x + np.random.multivariate_normal(np.zeros(n), 0.0001 * np.eye(n))[:, None]
            else:
                u = -Gain @ x

            if t == time_horizon:
                break

            # System transition
            x_prime = A.numpy() @ x + B.numpy() @ u + np.random.multivariate_normal(np.zeros(n), np.eye(n))[:, None]
            state = torch.from_numpy(np.concatenate((x, u, x_prime), axis=0)).unsqueeze(0)
            data = torch.cat((data, state), dim=0)
            x = x_prime

            t += 1

        k += 1
        if t == time_horizon:
            break

if naive:
    np.savetxt('gaussian_naive-iter.csv', ULA_iter.T, delimiter=',')
else:
    np.savetxt('gaussian_precond-iter.csv', ULA_iter.T, delimiter=',')