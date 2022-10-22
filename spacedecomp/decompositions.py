import torch
import numpy as np
from torch import optim
from tqdm.autonotebook import tqdm
from sklearn.decomposition import NMF


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Factorization:
    def __init__(self, data, type='FA', laten_dim=2, train_loadings=False, use_nmf=False, use_svd=False):
        self.N, self.J= data.shape
        self.L = laten_dim
        self.nonnegative = False

        if type=='PNMF':
            self.nonnegative = True

        if type=='FA':
            self.nonnegative = False

        self.use_nmf = use_nmf
        self.use_svd = use_svd
        #initiate pytorch variables
        sig_dim = self.N if self.nonnegative else self.J
            
        self.y_sig = torch.tensor(np.random.rand(sig_dim), dtype=torch.float, requires_grad= True, device=device)

        self.params = [self.y_sig]

        self.initiate_W(data, requires_grad=train_loadings)

    
    def initiate_W(self, Y: np.ndarray, requires_grad=False):

        if self.use_nmf:
            model = NMF(n_components=self.L, init='random', random_state=97)
            W = model.fit_transform(Y.T)
            F = None

        elif self.use_svd:
            C = np.dot(Y.T, Y)
            V, S2, _ = np.linalg.svd(C)
            S = np.sqrt(S2/self.N)
            F = (Y@V[:, :self.L])/S[:self.L]
            W = V[:,:self.L]*S[:self.L]
        else:
            W = np.random.randn(self.J, self.L)
            F = None

        self.W = torch.tensor(W, dtype=torch.float, requires_grad=requires_grad, device=device)
        self.initiate_F(F_init=F)


        if requires_grad:
            self.params.append(self.W)

    def initiate_F(self, F_init=None):

        if F_init is None:
            F_init = np.random.rand(self.N, self.L)
        
        f_sig = torch.tensor(np.ones(self.L), dtype=torch.float, device=device)
        f_mean = torch.tensor(np.zeros(self.L), dtype=torch.float, device=device)
        self.p_f = torch.distributions.Normal(f_mean, f_sig)

        self.f_qsig = torch.tensor(np.random.rand(self.N, self.L), dtype=torch.float, requires_grad=True, device=device)
        self.f_qmean = torch.tensor(F_init, dtype=torch.float, requires_grad=True, device=device)
        self.params.append(self.f_qsig)
        self.params.append(self.f_qmean)


    def train(self, data, epochs=500, lr=1e-2):
        data = data.to(device)
        optimizer= optim.Adam(self.params, lr=lr)
        running_loss = []
        for _ in tqdm(range(epochs)):

            optimizer.zero_grad()
            self.q_f = torch.distributions.Normal(self.f_qmean, torch.abs(self.f_qsig))
            y_mu = self.predict_mean()

            p_y = self.probability(y_mu)

            E = p_y.log_prob(data).sum()

            KL = torch.distributions.kl_divergence(self.q_f, self.p_f).sum()

            elbo = E-KL
            loss = -elbo/self.N
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.item())
        
        print('Finished training')

        return running_loss

    def predict_mean(self):
        F = self.q_f.rsample()

        if self.nonnegative:
            return torch.exp(F)@torch.abs(self.W.T)

        return F@(self.W.T)

    def probability(self, mean):

        if self.nonnegative:
            weighted_mean = torch.abs(self.y_sig) * (mean.T)
            return  torch.distributions.Poisson(weighted_mean.T)
        
        return torch.distributions.Normal(mean, torch.abs(self.y_sig))
