import torch
import numpy as np
from scipy.spatial import distance_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RBF:

    def __init__(self, X: np.ndarray = None, variance = None, lengthscale = None):


        self.sigma = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if variance is None else torch.tensor([variance**0.5], device=device, requires_grad=True, dtype=torch.float)
        self.lengthscale = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if lengthscale is None else torch.tensor([lengthscale], device=device, requires_grad=True, dtype=torch.float)
        self.X = X
        self.distance = self.build_distance_mat(self.X, self.X) if self.X else None

    def build_distance_mat(self, X: np.ndarray, Y: np.ndarray):

        return torch.tensor(distance_matrix(X, Y), dtype=torch.float, device=device)


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
        
        

        return (self.sigma**2) * torch.exp(-0.5 * (distance/self.lengthscale)**2)



    def params(self):
        return [self.sigma, self.lengthscale]

    def __call__(self, X : np.ndarray = None):

        if self.X is None:
            self.X = X
            self.distance = self.build_distance_mat(self.X, self.X)

        
        return self.predict()


class Matern:

    def __init__(self, X: np.ndarray = None, variance = None, lengthscale = None):


        self.sigma = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if variance is None else torch.tensor([variance**0.5], device=device, requires_grad=True, dtype=torch.float)
        self.lengthscale = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if lengthscale is None else torch.tensor([lengthscale], device=device, requires_grad=True, dtype=torch.float)
        self.X = X
        self.distance = self.build_distance_mat(self.X, self.X) if self.X else None

    def build_distance_mat(self, X: np.ndarray, Y: np.ndarray):

        return torch.tensor(distance_matrix(X, Y, p=1), dtype=torch.float, device=device)


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
        
        
        D = (3**0.5)*distance/torch.abs(self.lengthscale)

        return (self.sigma**2) * (1+D)* torch.exp(-D)



    def params(self):
        return [self.sigma, self.lengthscale]

    def __call__(self, X : np.ndarray = None):

        if self.X is None:
            self.X = X
            self.distance = self.build_distance_mat(self.X, self.X)

        
        return self.predict()



class Periodic:

    def __init__(self, X: np.ndarray = None, variance = None, lengthscale = None, period = None):


        self.sigma = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if variance is None else torch.tensor([variance**0.5], device=device, requires_grad=True, dtype=torch.float)
        self.lengthscale = torch.tensor(np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if lengthscale is None else torch.tensor([lengthscale], device=device, requires_grad=True, dtype=torch.float)
        self.X = X
        self.distance = self.build_distance_mat(self.X, self.X) if self.X else None
        self.period = torch.tensor(2*np.pi*np.random.rand(1,1), dtype=torch.float, device=device, requires_grad= True) if period is None else torch.tensor([period], device=device, requires_grad=True, dtype=torch.float)

    def build_distance_mat(self, X: np.ndarray, Y: np.ndarray):

        return torch.tensor(distance_matrix(X, Y, p=1), dtype=torch.float, device=device)


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
        
        
        D = torch.sin((np.pi)*distance/torch.abs(self.period))/self.lengthscale

        return (self.sigma**2) * torch.exp(-2*(D**2))



    def params(self):
        return [self.sigma, self.lengthscale, self.period]

    def __call__(self, X : np.ndarray = None):

        if self.X is None:
            self.X = X
            self.distance = self.build_distance_mat(self.X, self.X)

        
        return self.predict()