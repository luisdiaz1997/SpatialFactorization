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

    def log_likelihood(self):
        return self().log_prob(self.y)

    def predict(self, Xtest, verbose=True):
        
        K11 = self.kernel() + (self.noise**2)*torch.eye(self.N)
        K12 = self.kernel.predict(Y=Xtest)
        K21 = K12.T
        K22 = self.kernel.predict(X=Xtest, Y=Xtest)
        mean = K21 @ torch.inverse(K11) @ (self.y[:, None])
        mean = torch.squeeze(mean)

        cov = K22 - (K21@torch.inverse(K11)@K12)

        if verbose:
            print('mean: ', mean.shape)
            print('covariance: ', cov.shape)
        

        return mean, cov
    
    def fit(self, optimizer, lr=0.005, epochs=1000):

        opt = optimizer(self.params(), lr=lr)
        running_loss = []
        for _ in range(epochs):  # loop over the dataset multiple times
        
            opt.zero_grad()
            J = -self.log_likelihood()
            J.backward()
            
            opt.step()
            opt.zero_grad()

            # print statistics
            running_loss.append(J.item())

        print('Finished Training')

        return running_loss




