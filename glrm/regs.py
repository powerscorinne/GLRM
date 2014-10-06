from numpy import unique, maximum, ones, sign, Inf, minimum, round, \
        expand_dims, median, zeros, hstack, vstack, argmax, array, arange, tile, ceil, nan
from warnings import warn
from numpy.linalg import norm
from utils import elementwise_shrinkage

""" Regularizer functions """

class norm1_reg(object):
    
    def __init__(self, nu):
        self.nu = nu
        def reg(V): return nu*norm(V, 1)
        def prox(V, lmbd): return elementwise_shrinkage(V, lmbd*nu)
        def subgrad(V): return nu*sign(V)
        self.reg = reg
        self.prox = prox
        self.subgrad = subgrad

    def __str__(self):
        return "nu \|V\|_1"

    def __call__(self, V):
        return self.reg(V)


class norm2sq_reg(object):

    def __init__(self, nu):
        self.nu = nu
        def reg(V): return nu*norm(V, 2)**2
        # XXX def prox(V, lmbd): return
        def subgrad(V): return 2*nu*V
        self.reg = reg
        self.subgrad = subgrad

    def __str__(self):
        return "\|X\|_2^2"

    def __call__(self, V):
        return self.reg(V)

class zero_reg(object):
        
    def __init__(self, *args, **kwargs): # args have no bearing
        def reg(V): return 0
        def prox(V, *args, **kwargs): return V
        def subgrad(V): return zeros(V.shape)
        self.reg = reg
        self.prox = prox
        self.subgrad = subgrad

    def __str__(self):
        return "0"

    def __call__(self, V):
        return self.reg(V)
