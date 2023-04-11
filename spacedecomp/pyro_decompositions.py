import torch
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch.nn import Parameter
from pyro.contrib import gp

class NSF(PyroModule):
    def __init__(self, X, y, 
