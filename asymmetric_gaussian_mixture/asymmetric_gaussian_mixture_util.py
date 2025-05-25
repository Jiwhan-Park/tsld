import time
import torch
from auxiliary import*

def asymmetric_gaussian_mixture_grad_log(phi, z_batch, x_prime_batch, mean, grad_log_b, grad_log_phi, grad_sum, w, dot_batch, const_batch, r, n, m, lam, gamma, eta):

    # Assume all inputs are PyTorch tensors
    # Transform parameters
    theta_T = tr_phi_to_theta(phi, n, m).T.contiguous()  # (n + m, n)

    # Compute residual
    torch.matmul(theta_T.unsqueeze(0), z_batch, out = w)
    w.mul_(-1)
    w.add_(x_prime_batch)

    # Compute gradient components
    torch.matmul(w.squeeze(), mean, out=dot_batch)
    dot_batch.mul_(1 / r)
    torch.exp(dot_batch, out=const_batch)
    const_batch.mul_((1 - gamma) * eta)
    const_batch.add_(gamma)
    torch.reciprocal(const_batch, out=const_batch)
    const_batch.sub_(1)
    const_batch.mul_(gamma)
    torch.matmul(mean.unsqueeze(0), const_batch.unsqueeze(-1), out=grad_log_b)
    grad_log_b.add_(w)
    grad_log_b.div_(r)

    # Kronecker product computation between column vectors
    torch.bmm(grad_log_b, z_batch.transpose(1, 2), out = grad_log_phi)

    # Final summation
    grad_sum.fill_(-0.5)
    grad_sum.add_(phi)
    grad_sum.mul_(-lam)
    grad_sum.add_(grad_log_phi.sum(dim = 0).view(-1, 1))

    return

def asymmetric_gaussian_mixture_hess_log(phi, z_batch, x_prime_batch, zzT_batch, mean, mmT_batch, hess_init, hess_sum_init, hess_log_phi, hess_sum, w, dot_batch, exp_batch, const_batch, kron_term, r, n, m, lam, gamma, eta):

    # Assume all inputs are PyTorch tensors
    batch_size = len(w)
    # Transform parameters
    theta_T = tr_phi_to_theta(phi, n, m).T.contiguous()

    # Compute residual
    torch.matmul(theta_T.unsqueeze(0), z_batch, out=w)
    w.mul_(-1)
    w.add_(x_prime_batch)

    # Compute Hessian components
    torch.matmul(w.squeeze(), mean, out=dot_batch)
    dot_batch.mul_(1 / r)
    torch.exp(dot_batch, out=exp_batch)
    exp_batch.mul_((1 - gamma) * eta)
    torch.add(exp_batch, gamma, out=const_batch)
    torch.square(const_batch, out=dot_batch)
    torch.div(exp_batch, dot_batch, out=const_batch)
    const_batch.mul_(-gamma / r**2)
    hess_log_phi.copy_(const_batch.unsqueeze(-1).expand(batch_size, n, n))
    hess_log_phi.mul_(mmT_batch)
    hess_log_phi.add_(hess_init)

    # Vectorized Kronecker product computation
    torch.mul(hess_log_phi.view(batch_size, n, 1, n, 1).expand(batch_size, n, n + m, n, n + m), zzT_batch.view(batch_size, 1, n + m, 1, n + m).expand(batch_size, n, n + m, n, n + m), out=kron_term)

    # Final summation
    hess_sum.copy_(hess_sum_init)
    hess_sum.sub_(kron_term.view(batch_size, (n + m) * n, (n + m) * n).sum(dim=0))

    return

def newton_method(data, mean, r, n, m, lam, gamma, eta, N, device):
    batch_size = len(data)
    dtype = data.dtype
    data_gpu = data.to(device)
    mean_gpu = mean.to(device)
    phi = torch.zeros((n + m) * n, 1, device=device, dtype=dtype)
    grad_log_b = torch.empty(len(data), n, 1, device=device, dtype=dtype)
    grad_log_phi = torch.empty(len(data), n, n + m, device=device, dtype=dtype)
    grad = torch.empty((n + m) * n, 1, device=device, dtype=dtype)
    w = torch.empty(len(data), n, 1, device=device, dtype=dtype)
    dot_batch = torch.empty(len(data), 1, device=device, dtype=dtype)
    exp_batch = torch.empty(len(data), 1, device=device, dtype=dtype)
    const_batch = torch.empty(len(data), 1, device=device, dtype=dtype)
    hess_log_phi = torch.empty(batch_size, n, n, device=device, dtype=dtype)
    hess = torch.empty((n + m) * n, (n + m) * n, device=device, dtype=dtype)
    hess_init = (1 / r) * torch.eye(n, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    hess_sum_init = -lam * torch.eye((n + m) * n, device=device, dtype=dtype)
    kron_term = torch.empty(batch_size, n, n + m, n, n + m, device=device, dtype=dtype)
    z_batch = data_gpu[:, :(n + m), :]
    x_prime_batch = data_gpu[:, (n + m):, :]
    zzT_batch = torch.bmm(z_batch, z_batch.transpose(1, 2))
    mmT_batch = (mean_gpu @ mean_gpu.T).expand(batch_size, n, n)
    for idx in range(N):
        asymmetric_gaussian_mixture_hess_log(phi, z_batch, x_prime_batch, zzT_batch, mean_gpu, mmT_batch, hess_init, hess_sum_init, hess_log_phi, hess, w,
                                  dot_batch, exp_batch, const_batch, kron_term, r, n, m, lam, gamma, eta)
        asymmetric_gaussian_mixture_grad_log(phi, z_batch, x_prime_batch, mean_gpu, grad_log_b, grad_log_phi, grad, w,
                                  dot_batch, const_batch, r, n, m, lam, gamma, eta)
        phi -= torch.linalg.solve(hess, grad)

    return phi