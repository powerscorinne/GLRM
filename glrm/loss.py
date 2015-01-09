from numpy.linalg import norm
from numpy import sign, maximum, ceil, minimum
from util import scale
import cvxpy as cp

"""
Abstract loss class and canonical loss functions.
"""

# Abstract Loss class
class Loss(object):
    # shape indicates how quickly it grows: 0 [flat], 1 [linear], 2 [quadratic+]
    def loss(self, A, X, Y): raise NotImplementedError("Override me!")
    def encode(self, A): return A # default
    def decode(self, A): return A # default

    def __init__(self, A, missing = [], T = False):
        if T: A = A.T
        unscaled_loss = self.loss

        @scale(A, missing, T)
        def loss(X, Y): return unscaled_loss(A, X, Y)
        self.loss = loss

        self.n = A.shape[1] # number of columns of A

    def __str__(self): return "GLRM Loss: override me!"
    def __call__(self, X, Y): return self.loss(X, Y)

# Canonical loss functions
class QuadraticLoss(Loss):
    def loss(self, A, X, Y): return cp.square(A - X*Y)/2.0 # matrix format!
    def __str__(self): return "quadratic loss"
# 
# class HuberLoss(Loss):
#     shape = 1
#     a = 1.0 # XXX does the value of `a' propagate if we update it?
#     def loss(self, A, X, Y): 
#         B = A - X.dot(Y)
#         return ((abs(B) <= self.a)*B**2 + \
#                 (abs(B) > self.a)*(2*abs(B) - self.a)*self.a)
#     def subgrad(self, A, X, Y, mask):
#         B = A - X.dot(Y)
#         return -X.T.dot((2*(abs(B) <= self.a)*B + \
#                 (abs(B) > self.a)*sign(B)*2*self.a)*mask)
#     def __str__(self): return "huber loss"
# 
# class FractionalLoss(Loss):
#     shape = 3
#     PRECISION = 1e-2
#     def loss(self, A, X, Y):
#         U = X.dot(Y)
#         U = maximum(U, self.PRECISION) # to avoid dividing by zero
#         return maximum((A - U)/U, (U - A)/A)
# 
#     def subgrad(self, A, X, Y, mask):
#         U = X.dot(Y)
#         U = maximum(U, self.PRECISION) # to avoid dividing by zero
#         return -X.T.dot(((1.0/A)*(U < A) + (-A/U**2)*(U >= A))*mask)
# 
#     def __str__(self): return "fractional loss"
# 
# class HingeLoss(Loss):
#     shape = 1
#     def loss(self, A, X, Y): return maximum((1 - A*X.dot(Y)), 0)
#     def subgrad(self, A, X, Y, mask): return -X.T.dot(A*((1 - A*X.dot(Y)) > 0)*mask)
#     def decode(self, A): return sign(A) # convert back to Boolean
#     def __str__(self): return "hinge loss"
# 
# 
# class OrdinalLoss(Loss):
#     shape = 1
#     Amax, Amin = 0, 0
#     def loss(self, A, X, Y): 
#         U = X.dot(Y)
#         self.Amin, self.Amax = A.min(), A.max()
#         return sum([maximum(U - a, 0)*(a >= A) + maximum(-U + a + 1, 0)*(a < A)
#             for a in range(int(self.Amin), int(self.Amax))])
#     def subgrad(self, A, X, Y, mask):
#         U = X.dot(Y)
#         B = (U < self.Amin)*(self.Amin - A) + (U > self.Amax)*(self.Amax - A) \
#                 + sign(U - A)*ceil(abs(U - A))*((U >= self.Amin) & (U <= self.Amax))
#         return X.T.dot(B*mask)
#     def decode(self, A): return maximum(minimum(A.round(), self.Amax), self.Amin)
#     def __str__(self): return "ordinal loss"
# 
