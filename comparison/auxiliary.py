import torch
import numpy as np

def tr_phi_to_theta(phi, n, m):
    return phi.view(n, n + m).T