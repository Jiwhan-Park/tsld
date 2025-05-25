import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from scipy import stats
from control import dare
from gaussian_mixture_util import *
from gaussian_mixture_ULA import *

torch.set_default_dtype(torch.float64)

# System parameters
n = 10
m = 10
simul = 100
record = 0
time_horizon = 2000
regret_ULA = np.zeros((simul, time_horizon))
traj = np.empty((0, time_horizon - 1, 2 * n + m))
model_error = np.zeros((simul, math.ceil((math.sqrt(8 * time_horizon + 5) - 3) / 2)))
rejection = np.zeros((simul, math.ceil((math.sqrt(8 * time_horizon + 5) - 3) / 2)))
time_simul = np.zeros(simul)

# System matrices
A = torch.tensor([[0.6, 0.6, 0.5, 0.0, 0.1, 0.4, 0.3, 0.3, 0.3, 0.4],
                  [0.3, 0.2, 0.6, 0.0, 0.1, 0.0, 0.2, 0.5, 0.2, 0.0],
                  [0.0, 0.6, 0.0, 0.3, 0.4, 0.0, 0.5, 0.4, 0.1, 0.3],
                  [0.4, 0.1, 0.5, 0.6, 0.6, 0.5, 0.1, 0.1, 0.6, 0.0],
                  [0.5, 0.1, 0.2, 0.0, 0.1, 0.1, 0.1, 0.0, 0.6, 0.4],
                  [0.1, 0.2, 0.2, 0.1, 0.2, 0.0, 0.5, 0.2, 0.5, 0.7],
                  [0.3, 0.6, 0.1, 0.6, 0.1, 0.0, 0.3, 0.4, 0.6, 0.3],
                  [0.3, 0.0, 0.5, 0.2, 0.2, 0.7, 0.4, 0.1, 0.4, 0.3],
                  [0.0, 0.3, 0.3, 0.5, 0.3, 0.5, 0.1, 0.0, 0.1, 0.5],
                  [0.3, 0.0, 0.0, 0.5, 0.0, 0.2, 0.4, 0.4, 0.0, 0.5]])
B = torch.tensor([[0.5, 0.4, 0.2, 0.5, 0.4, 0.0, 0.8, 0.1, 0.3, 0.7],
                  [0.1, 0.4, 0.6, 0.0, 0.5, 0.0, 0.3, 0.1, 0.3, 0.2],
                  [0.0, 0.5, 0.0, 0.6, 0.6, 0.5, 0.0, 0.0, 0.1, 0.2],
                  [0.4, 0.4, 0.3, 0.5, 0.0, 0.1, 0.5, 0.0, 0.2, 0.4],
                  [0.2, 0.1, 0.4, 0.0, 0.0, 0.7, 0.1, 0.1, 0.5, 0.3],
                  [0.4, 0.5, 0.0, 0.6, 0.0, 0.4, 0.6, 0.1, 0.4, 0.5],
                  [0.3, 0.5, 0.0, 0.3, 0.1, 0.7, 0.2, 0.0, 0.4, 0.6],
                  [0.2, 0.0, 0.1, 0.6, 0.2, 0.7, 0.0, 0.1, 0.4, 0.4],
                  [0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.3, 0.1, 0.4],
                  [0.2, 0.5, 0.1, 0.3, 0.0, 0.5, 0.4, 0.4, 0.2, 0.3]])
Q = 2 * np.eye(n)
R = np.eye(m)

# Admissible conditions
theta_bound = 20
J_bound = 20000
rho = 0.99

# Initial prior
lam = 10

# Noise parameters
gm_a = np.array([1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4])
r = 1
mu = np.array([gm_a, -gm_a])
cov = np.array([r * np.eye(n), r * np.eye(n)])
weights = np.array([0.5, 0.5])
W = r * np.eye(n) + gm_a[:, None] @ gm_a[:, None].T

# Convert gm_a from NumPy to PyTorch
gm_a = torch.from_numpy(gm_a).view(-1, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for simulation in range(simul):
    print(f"Simulation {simulation + 1}")
    restart_sim = False
    t = 1
    t_k = 0
    k = 1
    x = np.zeros((n, 1))
    data = torch.empty(0, 2 * n + m, 1)
    sim_start = time.time()
    while True:
        print(f"Time step {t}")
        T = k + 1
        t_k = t

        # Parameter estimation
        phi_U = newton_method(data, gm_a, r, n, m, lam, 1000, device) # ((n + m) * n, 1)

        # Policy optimization with rejection sampling
        while True:
            start = time.time()
            arbi = gaussian_mixture_ULA(gm_a, r, lam, phi_U, data, n, m, t_k, device)
            end = time.time()
            print(f"ULA total time: {end - start:.4e}")
            # print(arbi)
            model_error[simulation, k - 1] = np.linalg.norm(arbi.T.numpy() - np.concatenate((A, B), axis=1)) / np.linalg.norm(np.concatenate((A, B), axis=1))

            a = arbi[:n, :].T.numpy()
            b = arbi[n:, :].T.numpy()

            # Solve discrete algebraic Riccati equation
            S, _, Gain = dare(a, b, Q, R)

            # Check stability conditions
            cond1 = np.linalg.norm(arbi) < theta_bound
            cond2 = np.trace(W @ S) < J_bound
            eigvals = np.linalg.eigvals((a - b @ Gain).T @ (a - b @ Gain))
            cond3 = np.sqrt(np.max(eigvals)) < rho

            if cond1 and cond2 and cond3:
                break
            print('Rejection')
            rejection[simulation][k - 1] += 1

        # System interaction loop
        while t <= t_k + T - 1:
            if t == t_k + T - 1:
                u = -Gain @ x + np.random.multivariate_normal(np.zeros(n), 0.0001 * np.eye(n))[:, None]
            else:
                u = -Gain @ x

            # Cost calculation
            cost = x.T @ Q @ x + u.T @ R @ u
            J = np.trace(W @ S)
            regret_ULA[simulation, t - 1] = cost[0, 0] - J

            if t == time_horizon:
                break

            # System transition
            idx = np.random.choice(2, p=weights)
            x_prime = A.numpy() @ x + B.numpy() @ u + np.random.multivariate_normal(mu[idx], cov[idx])[:, None]
            state = torch.from_numpy(np.concatenate((x, u, x_prime), axis=0)).unsqueeze(0)
            data = torch.cat((data, state), dim=0)
            x = x_prime
            t += 1

        k += 1
        if t == time_horizon:
            traj = np.concatenate((traj, np.expand_dims(data.numpy().squeeze(), axis=0)), axis=0)
            break
    sim_end = time.time()
    time_simul[simulation] = sim_end - sim_start

# Save and plot results
cumreg_ULA = np.cumsum(regret_ULA, axis=1)
np.savetxt('10D-gaussian_mixture-regret.csv', cumreg_ULA.T, delimiter=',')
np.savetxt('10D-gaussian_mixture-moderr.csv', model_error.T, delimiter=',')
np.savetxt('10D-gaussian_mixture-rejection.csv', rejection.T, delimiter=',')
np.savetxt('10D-gaussian_mixture-time.csv', time_simul.T, delimiter=',')
for simulation in range(simul):
    np.savetxt(f'10D-gaussian_mixture-traj{simulation + record}.csv', traj[simulation], delimiter=',')

plt.figure()
x_vals = np.arange(1, time_horizon + 1)
y = np.mean(cumreg_ULA, axis=0)

# Confidence interval calculation
t_crit = stats.t.ppf([0.025, 0.975], simul - 1)
sem = stats.sem(cumreg_ULA, axis=0)
lower = y + t_crit[0] * sem
upper = y + t_crit[1] * sem

plt.fill_between(x_vals, lower, upper, alpha=0.2)
plt.plot(x_vals, y, 'r-', label='Symmetric GM 10D')
plt.xlabel('$T$', fontsize=16)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.legend()
plt.show()