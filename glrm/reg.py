from numpy.linalg import norm
from numpy import sign, Inf
from util import shrinkage

"""
Abstract reg class and canonical regularizer functions.
"""

# Abstract Reg class
class Reg(object):
    # shape indicates how quickly it grows: 0 [flat], 1 [linear], 2 [quadratic+]
    def reg(self, X): raise NotImplementedError("Override me!")
    def subgrad(self, X): raise NotImplementedError("Override me!")
    def prox(self, X, lmbd): raise NotImplementedError("Override me!")

    def __init__(self, nu=1): self.nu = nu # XXX think of a better way to handle nu?
    def __str__(self): return "GLRM Reg: override me!"
    def __call__(self, X): return self.reg(X)

class ZeroReg(Reg):
    shape = 0
    def reg(self, X): return 0
    def subgrad(self, X): return 0
    def prox(self, X, lmbd): return X
    def __str__(self): return "zero reg"

class LinearReg(Reg):
    shape = 1
    def reg(self, X): return self.nu*norm(X, 1)
    def subgrad(self, X): return self.nu*sign(X)
    def prox(self, X, lmbd): return self.nu*shinkage(X, lmbd*self.nu)
    def __str__(self): return "linear reg"

class QuadraticReg(Reg):
    shape = 2
    def reg(self, X): return self.nu*norm(X, 2)**2
    def subgrad(self, X): return 2*self.nu*X
    def prox(self, X, lmbd): return 1.0/(1+lmbd*self.nu)*X
    def __str__(self): return "quadratic reg"

class NonnegativeReg(Reg):
    shape = 0
    def reg(self, X): 
        if (X < 0).sum() > 0: return Inf
        else: return 0
    def subgrad(self, X): raise Exception("no subgradient for nonnegative reg")
    def prox(self, X, lmbd): return X*(X>0) 

