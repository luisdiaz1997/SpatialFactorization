import torch
import numpy as np
from scipy.spatial import distance_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RBF:

    def __init__(self):

        self.sigma = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True)
        self.lengthscale = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True)
        self.distance = None

    def build_distance_mat(self, X: np.ndarray):
        self.distance = torch.tensor(distance_matrix(X, X), dtype=torch.float)

    def __call__(self, X = None):

        if X is not None:
            self.build_distance_mat(X)

        return (self.sigma**2) * torch.exp(-0.5 * (self.distance/self.lengthscale)**2)