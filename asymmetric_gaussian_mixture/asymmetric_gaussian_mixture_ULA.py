import torch
import math
import time
from auxiliary import *
from asymmetric_gaussian_mixture_util import *

def asymmetric_gaussian_mixture_ULA(mean, r, lam, gamma, eta, phi_U, data, n, m, t_k, device):

    # Copy to GPU
    dtype = phi_U.dtype
    mean_gpu = mean.to(device)
    phi_gpu = phi_U.to(device)
    data_gpu = data.to(device)

    # Batch data processing
    z_batch = data_gpu[:, :(n + m), :]       # (batch_size, n + m, 1)
    x_prime_batch = data_gpu[:, (n + m):, :] # (batch_size, n, 1)

    # Preconditioner calculation (parallel implementation)
    preconditioner = lam * torch.eye((n + m) * n, device = device, dtype = dtype)
    zzT = torch.bmm(z_batch, z_batch.transpose(1, 2))  # (batch_size, n + m, n + m)
    zzT = zzT.sum(dim = 0).repeat(n, 1, 1)             # (n, n + m, n + m)
    preconditioner += torch.block_diag(*zzT)           # ((n + m) * n, (n + m) * n)
    inv_preconditioner = torch.linalg.solve(preconditioner, torch.eye((n + m) * n, device = device, dtype = dtype))

    # Strong log-concavity and Lipschitz smoothness
    M = 1 / r * (1 - 1 / (4 * r) * (mean.T @ mean).item())
    L = 1 / r

    # Eigenvalue calculations
    eigvals = torch.linalg.eigvalsh(preconditioner)
    min_eig = eigvals[0]

    # Step size calculation
    step_size = (M * min_eig) / (16 * L**2 * max(t_k, min_eig))
    step_iter = max(1, math.ceil(4 * math.log2(max(t_k, min_eig) / min_eig) / (M * step_size)))
    print('step_iter:', step_iter)

    cholesky_inv_precond = torch.linalg.cholesky(2 * step_size * inv_preconditioner)

    grad_log_b = torch.empty(len(data), n, 1, device=device, dtype=dtype)
    grad_log_phi = torch.empty(len(data), n, n + m, device=device, dtype=dtype)
    grad_U = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    scaled_grad_U = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    w = torch.empty(len(data), n, 1, device=device, dtype=dtype)
    noise = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    scaled_noise = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    dot_batch = torch.empty(len(data), 1, device=device, dtype=dtype)
    const_batch = torch.empty(len(data), 1, device=device, dtype=dtype)

    # Main ULA loop
    for step in range(step_iter):
        # t1 = time.time()
        # Compute gradient of U
        asymmetric_gaussian_mixture_grad_log(phi_gpu, z_batch, x_prime_batch, mean_gpu, grad_log_b, grad_log_phi, grad_U, w,
                                             dot_batch, const_batch, r, n, m, lam, gamma, eta)
        grad_U.mul_(-1)
        # t2 = time.time()
        # Preconditioned gradient step
        torch.matmul(inv_preconditioner, grad_U, out = scaled_grad_U)
        # t3 = time.time()
        # Noise injection with structural covariance
        noise.normal_(mean = 0, std = 1)
        # t4 = time.time()
        torch.matmul(cholesky_inv_precond, noise, out = scaled_noise)
        # t5 = time.time()
        phi_gpu.sub_(step_size * scaled_grad_U)
        phi_gpu.add_(scaled_noise)
        # t6 = time.time()
        # if (step % 100000 == 0):
        #     print(f'iteration {step + 1:7d}: {t6 - t1:.3e}')
        #     print(f'time: {t2 - t1:.3e}, {t3 - t2:.3e}, {t4 - t3:.3e}, {t5 - t4:.3e}, {t6 - t5:.3e}')

    return tr_phi_to_theta(phi_gpu, n, m).to("cpu")