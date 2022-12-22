import torch
import numpy as np
from torch import distributions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GP:

    def __init__(self, X, y, kernel, noise=0.1):
        
        self.y = y
        self.kernel = kernel
        self.noise = torch.tensor(noise, dtype=torch.float, device=device, requires_grad= True)
        self.X = X
        self.N = self.X.shape[0]
        self.mean = torch.zeros(self.N, dtype=torch.float, device=device)

    def params(self):
        return [self.noise] + self.kernel.params()
        
    def __call__(self):
        self.covariance = self.kernel(self.X) + (self.noise**2)*torch.eye(self.N)
        return distributions.MultivariateNormal(self.mean, self.covariance)