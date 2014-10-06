from math import sqrt
from numpy import Inf, hstack, vstack, ones, trace, zeros
from numpy.linalg import norm
from warnings import warn
from copy import copy

"""
Block object handles an embedded optimization problem
(and solves it using gradient descent). 
Blocks are initialized by a generic loss function, e.g., squared_loss.
Use the 'update' interface to update X, Y, i.e.,

    # update Y
    Y = my_block.update(X, Y)

Input is P (parameter) and V (Variable) 
which either take the form X, Y or Y.T, X.T.
(Or arrays of these.)
Block objects *can* handle multiple loss functions 
(losses is an array) and corresponingly, multiple parameters P
(Ps is an array of parameters).
This is necessary for solving samples (X update) across multiple features.
"""
# XXX fix gradient descent parameter

# XXX block only supports one reg, although can handle a list of losses (for
# optimizing over X)

class Block(object):

    def __init__(self, losses, reg=None):
        if not isinstance(losses, list): losses = [losses]
        self.L = losses
        self.r = reg
        self.alg = "grad"

    def __str__(self):
        return "{0} + {1}".format(str(self.L), str(self.r))

    def __call__(self, Ps, V):
        # evalulate losses in this block with parameter P and variable V
        # does *not* include regularizer
        if not isinstance(Ps, list): Ps = [Ps]
        return sum([L(P, V) for L, P in zip(self.L, Ps)])

    def update(self, Ps, V_warm, RELTOL, max_iters, skip_last_col = False):
        # minimize over V with P fixed
        if not isinstance(Ps, list): Ps = [Ps]
        # call an algorithm below
        self._alg(Ps, V_warm, RELTOL, max_iters, skip_last_col)

    @property
    def alg(self):
        if self._alg == self._accl_prox_grad_desc: return "accl proximal gradient descent"
        elif self._alg == self._gradient_descent: return "gradient descent (ignores regularizers)"
        else: return "unknown"

    @alg.setter
    def alg(self, alg_str):
        if "prox" in alg_str.lower(): self._alg = self._accl_prox_grad_desc
        # XXX add a check that have no regularizers if want to use grad desc
        elif "grad" in alg_str.lower(): self._alg = self._gradient_descent
        else: warn("\n '" + alg_str + "' not supported.")

    # ============== Solvers for embedded optimization problem ===================

    def _accl_prox_grad_desc(self, Ps, V_warm, RELTOL, max_iters, skip_last_col):
        # XXX need to implement skip_last_col
        V = V_warm # passed by reference; all updates to V propagate back
        lmbd, beta, k = 0.5, 0.5, 1.0 # alg parameters
        V0 = zeros(V.shape)
        f = Inf
        while True: # not converged
            f0 = f
            W = V + k/(k+3)*(V - V0)
            V0 = copy(V)
            while True: # line search
                Z = self.r.prox(W - lmbd*sum([L.subgrad(P, W) for L, P in
                    zip(self.L, Ps)]), lmbd)
                if self(Ps, Z) <= self._f_hat(Ps, Z, W, lmbd): break
                lmbd = lmbd*beta
            V = Z
            
            # check convergence
            f = self(Ps, V)
            if abs(f - f0) < RELTOL*(Ps[0].shape[0])*V.shape[1]: break
            k += 1

    def _f_hat(self, Ps, V1, V2, lmbd): # used for line search
        V_diff = V1 - V2
        V_subgrad = sum([L.subgrad(P, V2) for L, P in zip(self.L, Ps)])
        return self(Ps, V2) + trace(V_subgrad.T.dot(V_diff)) + 1/(2.0*lmbd)*trace(V_diff.T.dot(V_diff))

    def _gradient_descent(self, Ps, V_warm, RELTOL, max_iters, skip_last_col):
        V = V_warm # passed by reference; all updates to V propagate back 
        alpha = 0.01/(V.shape[0]*V.shape[1]) # XXX pick better alpha_1
        f = Inf
        for i in range(max_iters):
            f0 = f
            
            # gradient step
            grad = sum([L.subgrad(P, V) for L, P in zip(self.L, Ps)])
            grad_r = self.r.subgrad(V)
            grad_r[-1:,:] = 0 # never regularize last row of V
            if skip_last_col: grad[-1:,:] = 0 # don't update last row of V
            
            V -= alpha/sqrt(i+1)*(grad + grad_r) 
            
            # check convergence
            f = self(Ps, V)
            if abs(f - f0) < RELTOL*Ps[0].shape[0]*V.shape[1]: break
            if i == max_iters-1: warn("hit max iters for gradient descent")
        
        if f > f0: warn("your alpha parameter in gradient descent needs tuning...")
