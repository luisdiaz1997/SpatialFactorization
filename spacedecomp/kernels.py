import torch
import numpy as np
from abc import ABC, abstractmethod

from scipy.spatial import distance_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Kernel(ABC):
    
    def __init__(self, X: np.ndarray = None, variance=None, lengthscale = None) -> None:

        self.sigma = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if variance is None else torch.tensor([variance**0.5], device=device, requires_grad=True, dtype=torch.float)
        self.lengthscale = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if lengthscale is None else torch.tensor([lengthscale], device=device, requires_grad=True, dtype=torch.float)
        self.X = X
        self.distance = self.build_distance_mat(self.X, self.X) if self.X else None
        self.parameters = [self.sigma, self.lengthscale]
    
    def build_distance_mat(self, X: np.ndarray, Y: np.ndarray) -> torch.Tensor:

        return torch.tensor(distance_matrix(X, Y, p=self.norm), dtype=torch.float, device=device)

    @property
    @abstractmethod
    def norm(self)-> int:
        pass

    @abstractmethod
    def build_kernel(self, distance)-> torch.Tensor:
        pass

    
    def params(self):
        return self.parameters


    def predict(self, X: np.ndarray = None, Y: np.ndarray = None):
        '''
        Uses distance matrix with the current data X or with unseen data Y to build kernel matrix.

        Parameters:
            X (np.ndarray): new training
            Y (np.ndarray): test data
        
        Returns:
            kernel (torch.Tensor): Kernel of K(X, X) or K(X, Y)
        '''

        if X is None:
            if self.X is None: #If X is still None
                raise RuntimeError('Failed to initialize training data, please initialize self.X or use custom X')
            
            X = self.X
            distance = self.distance if Y is None else self.build_distance_mat(X, Y)
        
        else:
            distance = self.build_distance_mat(X, X) if Y is None else self.build_distance_mat(X, Y)


        return self.build_kernel(distance)
    

    def __call__(self, X : np.ndarray = None):

        if self.X is None:
            self.X = X
            self.distance = self.build_distance_mat(self.X, self.X)

        return self.predict()


class RBF(Kernel):

    def __init__(self, X: np.ndarray = None, variance = None, lengthscale = None):

        super(RBF, self).__init__(X, variance, lengthscale)
        self.p = 2

    @property
    def norm(self):
        return self.p

    def build_kernel(self, distance: torch.Tensor) -> torch.Tensor:
        return (self.sigma**2) * torch.exp(-0.5 * (distance/self.lengthscale)**2)

class Matern(Kernel):

    def __init__(self, X: np.ndarray = None, variance = None, lengthscale = None):


        super(Matern, self).__init__(X, variance, lengthscale)
        self.p = 1
    
    @property
    def norm(self):
        return self.p

    def build_kernel(self, distance: torch.Tensor) -> torch.Tensor:
        
        D = (3**0.5)*distance/torch.abs(self.lengthscale)
        return (self.sigma**2) * (1+D)* torch.exp(-D)



class Periodic(Kernel):

    def __init__(self, X: np.ndarray = None, variance = None, lengthscale = None, period = None):


        super(Periodic, self).__init__(X, variance, lengthscale)
        self.p = 1
        self.period = torch.tensor(2*np.pi*np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if period is None else torch.tensor([period], device=device, requires_grad=True, dtype=torch.float)
        self.parameters.append(self.period)

    @property
    def norm(self):
        return self.p
    
    def build_kernel(self, distance) -> torch.Tensor:

        D = torch.sin((np.pi)*distance/torch.abs(self.period))/self.lengthscale

        return (self.sigma**2) * torch.exp(-2*(D**2))        