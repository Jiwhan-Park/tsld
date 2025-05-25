# import time
import torch
import math
from auxiliary import *

def gaussian_ULA(lam, theta_mean_, data, n, m, t_k, device, naive=False):

    # Copy to GPU
    dtype = theta_mean_.dtype
    theta_gpu = theta_mean_.to(device)
    data_gpu = data.to(device)

    # Batch data processing
    z_batch = data_gpu[:, :(n + m), :]       # (batch_size, n + m, 1)
    x_prime_batch = data_gpu[:, (n + m):, :] # (batch_size, n, 1)

    # Preconditioner calculation
    preconditioner = lam * torch.eye((n + m) * n, device = device, dtype = dtype)
    zzT = torch.bmm(z_batch, z_batch.transpose(1, 2))  # (batch_size, n + m, n + m)
    zzT = zzT.sum(dim = 0).repeat(n, 1, 1)             # (n, n + m, n + m)
    preconditioner += torch.block_diag(*zzT)           # ((n + m) * n, (n + m) * n)
    inv_preconditioner = torch.linalg.solve(preconditioner, torch.eye((n + m) * n, device = device, dtype = dtype))

    # Strong log-concavity and Lipschitz smoothness parameters
    M = 1
    L = 1

    # Eigenvalue calculations
    eigvals = torch.linalg.eigvalsh(preconditioner)
    min_eig = eigvals[0]
    max_eig = eigvals[-1]
    print(f'norm={torch.linalg.norm(inv_preconditioner):.3e}, min_eig={min_eig:.3e}')

    # Step size calculation
    step_size = (M * min_eig) / (16 * L**2 * max(t_k, min_eig))
    # step_iter = max(1, math.ceil(4 * math.log2(max(t_k, min_eig) / min_eig) / (M * step_size)))
    step_iter = math.ceil(4 * math.log2(max(t_k, min_eig) / min_eig) / (M * step_size))
    if naive:
        step_size = (M * min_eig) / (16 * L**2 * max_eig**2)
        step_iter = math.ceil(64 * max_eig**2 / min_eig**2)
    print('step_iter:', step_iter)

    cholesky_inv_precond = torch.linalg.cholesky(2 * step_size * inv_preconditioner)

    if naive:
        inv_preconditioner = torch.eye((n + m) * n, device=device, dtype=dtype)

    theta_mean = lam * theta_gpu
    precision = lam * torch.eye(n + m, device=device, dtype=dtype)
    precision.add_(torch.sum(torch.bmm(z_batch, z_batch.transpose(1, 2)), dim=0))
    sigma = torch.linalg.solve(precision, torch.eye(n + m, device=device, dtype=dtype))
    zxpT = torch.sum(torch.bmm(z_batch, x_prime_batch.transpose(1, 2)), dim=0)
    theta_mean.add_(zxpT)
    theta_mean = sigma @ theta_mean
    phi = theta_mean.T.reshape((n + m) * n, 1)

    s1 = -torch.block_diag(*zzT)
    s2 = zxpT.T.reshape((n + m) * n, 1)
    s_prior = -lam * (phi - (theta_mean_.to(device)).T.reshape((n + m) * n, 1))

    grad_U = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    scaled_grad_U = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    noise = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    scaled_noise = torch.empty((n + m) * n, 1, device=device, dtype=dtype)

    # Main ULA loop
    for step in range(step_iter):
        # t1 = time.time()
        torch.matmul(s1, phi, out=grad_U)
        grad_U.add_(s2)
        grad_U.mul_(-1)
        grad_U.sub_(s_prior)
        # t2 = time.time()
        # Preconditioned gradient step
        torch.matmul(inv_preconditioner, grad_U, out = scaled_grad_U)
        # t3 = time.time()
        # System noise injection
        noise.normal_(mean = 0, std = 1)
        # t4 = time.time()
        torch.matmul(cholesky_inv_precond, noise, out = scaled_noise)
        # t5 = time.time()
        phi.sub_(step_size * scaled_grad_U)
        phi.add_(scaled_noise)
        # t6 = time.time()
        # if (step % 100000 == 0):
        #     print(f'iteration {step + 1:7d}: {t6 - t1:.3e}')
        #     print(f'time: {t2 - t1:.3e}, {t3 - t2:.3e}, {t4 - t3:.3e}, {t5 - t4:.3e}, {t6 - t5:.3e}')

    return step_iter, tr_phi_to_theta(phi, n, m).to("cpu")