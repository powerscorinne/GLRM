from numpy.linalg import norm
from numpy import sign

"""
Abstract reg class and canonical regularizer functions.
"""

# Abstract Reg class
class Reg(object):
    def reg(self, X): raise NotImplementedError("Override me!")
    def subgrad(self, X): raise NotImplementedError("Override me!")

    def __init__(self, nu=1): self.nu = nu # XXX think of a better way to handle nu?
    def __str__(self): return "GLRM Reg: override me!"
    def __call__(self, X): return self.reg(X)

class ZeroReg(Reg):
    def reg(self, X): return 0
    def subgrad(self, X): return 0
    def __str__(self): return "zero reg"

class LinearReg(Reg):
    def reg(self, X): return self.nu*norm(X, 1)
    def subgrad(self, X): return self.nu*sign(X)
    def __str__(self): return "linear reg"

class QuadraticReg(Reg):
    def reg(self, X): return self.nu*norm(X, 2)**2
    def subgrad(self, X): return 2*self.nu*X
    def __str__(self): return "quadratic reg"    
