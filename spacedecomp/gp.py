import torch
import numpy as np
from torch import distributions

from .kernels import RBF

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GP:

    def __init__(self, X, mean, kernel: RBF, noise=0.1):
        
        self.mean = mean
        self.kernel = kernel
        self.noise = np.array([noise])
        self.X = X
        self.N = X.shape[0]

        self.initialize()
    
    def initialize(self):
        self.sigma_noise = torch.tensor(self.noise, dtype=torch.float, device=device, requires_grad= True)
        self.kernel.build_distance_mat(self.X)
    
    def params(self):
        return [self.sigma_noise, self.kernel.sigma, self.kernel.lengthscale]

    def __call__(self, X=None):
        self.covariance = self.kernel(X) + (self.sigma_noise**2)*torch.eye(self.N)
        return distributions.MultivariateNormal(self.mean, self.covariance)