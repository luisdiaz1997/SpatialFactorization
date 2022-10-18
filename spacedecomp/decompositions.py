import torch
import numpy as np
from torch import optim
from tqdm.autonotebook import tqdm
from sklearn.decomposition import NMF



class FA:
    def __init__(self, data, laten_dim=2, train_loadings=False):
        self.N, self.J= data.shape
        self.L = laten_dim
        

        #initiate pytorch variables
        self.y_sig = torch.tensor(np.random.rand(self.J), dtype=torch.float, requires_grad= True)

        self.params = [self.y_sig]

        self.initiate_W(data, requires_grad=train_loadings)
        self.initiate_F()


    
    def initiate_W(self, Y: np.ndarray, requires_grad=False):

        C = np.dot(Y.T, Y)
        V, _, _ = np.linalg.svd(C)
        V = V*np.sqrt(self.J)
        self.W = torch.tensor(V[:,:self.L], dtype=torch.float, requires_grad=requires_grad)
        if requires_grad:
            self.params.append(self.W)

    def initiate_F(self):
        
        f_sig = torch.tensor(np.ones(self.L), dtype=torch.float)
        f_mean = torch.tensor(np.zeros(self.L), dtype=torch.float)
        self.p_f = torch.distributions.Normal(f_mean, f_sig)

        self.f_qsig = torch.tensor(np.random.rand(self.N, self.L), dtype=torch.float, requires_grad=True)
        self.f_qmean = torch.tensor(np.random.randn(self.N, self.L), dtype=torch.float, requires_grad=True)
        self.params.append(self.f_qsig)
        self.params.append(self.f_qmean)

    def get_factors(self):
        q_f = torch.distributions.Normal(self.f_qmean, torch.abs(self.f_qsig))
        F = q_f.rsample()
        return F.detach().numpy()

    def train(self, data, epochs=500, lr=1e-2):
        optimizer= optim.Adam(self.params, lr=lr)
        running_loss = []
        for _ in tqdm(range(epochs)):

            optimizer.zero_grad()
            q_f = torch.distributions.Normal(self.f_qmean, torch.abs(self.f_qsig))
            F = q_f.rsample()
            y_mu = F @ (self.W.T)

            p_y = torch.distributions.Normal(y_mu, torch.abs(self.y_sig))

            E = p_y.log_prob(data).sum(axis=1)

            KL = torch.distributions.kl_divergence(q_f, self.p_f).sum(axis=1)

            elbo = torch.mean(E)-torch.mean(KL)
            loss = -elbo
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.item())
        
        print('Finished training')

        return running_loss


class PNMF:
    def __init__(self, data, laten_dim=2, train_loadings=False):
        self.N, self.J= data.shape
        self.L = laten_dim
        

        #initiate pytorch variables
        self.y_sig = torch.tensor(np.random.rand(self.N), dtype=torch.float, requires_grad= True)

        self.params = [self.y_sig]

        self.initiate_W(data, requires_grad=train_loadings)
        self.initiate_F()


    
    def initiate_W(self, Y: np.ndarray, requires_grad=False):

        model = NMF(n_components=self.L, init='random', random_state=0)
        W = model.fit_transform(Y.T)
        
        self.W = torch.tensor(W, dtype=torch.float, requires_grad=requires_grad)
        if requires_grad:
            self.params.append(self.W)

    def initiate_F(self):
        
        f_sig = torch.tensor(np.ones(self.L), dtype=torch.float)
        f_mean = torch.tensor(np.zeros(self.L), dtype=torch.float)
        self.p_f = torch.distributions.Normal(f_mean, f_sig)

        self.f_qsig = torch.tensor(np.random.rand(self.N, self.L), dtype=torch.float, requires_grad=True)
        self.f_qmean = torch.tensor(np.random.randn(self.N, self.L), dtype=torch.float, requires_grad=True)
        self.params.append(self.f_qsig)
        self.params.append(self.f_qmean)

    def get_factors(self):
        q_f = torch.distributions.Normal(self.f_qmean, torch.abs(self.f_qsig))
        F = q_f.rsample()
        return F.detach().numpy()

    def train(self, data, epochs=500, lr=1e-2):
        optimizer= optim.Adam(self.params, lr=lr)
        running_loss = []
        for _ in tqdm(range(epochs)):

            optimizer.zero_grad()
            q_f = torch.distributions.Normal(self.f_qmean, torch.abs(self.f_qsig))
            F = q_f.rsample()
            y_mu = torch.exp(F) @ (torch.abs(self.W).T)

            p_y = torch.distributions.Poisson((torch.abs(self.y_sig) * y_mu.T).T)

            E = p_y.log_prob(data).sum(axis=1)

            KL = torch.distributions.kl_divergence(q_f, self.p_f).sum(axis=1)

            elbo = torch.mean(E)-torch.mean(KL)
            loss = -elbo
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.item())
        
        print('Finished training')

        return running_loss
    

