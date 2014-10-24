from convergence import Convergence
from numpy.linalg import norm
from math import sqrt
import time
from numpy import trace, zeros
from copy import copy

class Algorithm(object):
    """ Abstract algorithm class. """ 
    def __init__(self): self.converge = Convergence()
    def __str__(self): return "GLRM algorithm"
    def minimize(self, X, Y):
        """ Iterate minimization until converges. """
        self.converge.reset()
        while not self.converge.d(): Z = self._update(X, Y)
        return Z # could be singleton (X or Y) or both
    def _update(self, X, Y): raise NotImplementedError("override me")


class SGD(Algorithm):
    """ Stochastic gradient descent. """
    
    def __init__(self, subgrad, stepsize, obj):
        super(SGD, self).__init__()
        self.subgrad = subgrad # function that takes X, Y, returns subgradient w.r.t. Y
        self.stepsize = stepsize
        self.obj = obj

    def _update(self, X, Y):
        """ 
        @X: parameter (fixed)
        @Y: current value of variable
        return: Y -= stepsize*subgradient(X, Y) (sgd w.r.t. Y)
        """
        Y -= self.stepsize/(len(self.converge)+1.0)*self.subgrad(X, Y)
        self.converge.val.append(Y)
        self.converge.obj.append(self.obj(X, Y))
        return Y

    def __str__(self): return "stochastic gradient descent for (inner loop of) GLRM"

class ProxGD(Algorithm):
    """ Proximal gradient descent. """

    def __init__(self, (subgrad, prox), lmbd, obj):
        super(ProxGD, self).__init__()
        self.subgrad = subgrad
        self.prox = prox
        self.lmbd = lmbd
        self.beta = 0.5
        self.obj = obj

    def _update(self, X, Y):
        """ 
        @X: parameter (fixed)
        @Y: current value of variable
        return: Y after ProxGD converges
        """
        try: Y0 = copy(self.converge.val[-2])
        except: Y0 = zeros(Y.shape)

        k = len(self.converge.obj)+1.0
        W = Y + k/float(k+3)*(Y-Y0)

        while True: # line search
            Z = self.prox(W - self.lmbd*self.subgrad(X, W), self.lmbd)
            if self.obj(X, Z) < self._f_hat(X, Z, W, self.lmbd): break
            self.lmbd  = self.lmbd*self.beta

        Y = copy(Z)
        self.converge.val.append(Y)
        self.converge.obj.append(self.obj(X, Y))
        return Y

    def _f_hat(self, X, Y1, Y2, lmbd): # used for line search
        Y_diff = Y1 - Y2
        return self.obj(X, Y2) + trace(self.subgrad(X, Y2).T.dot(Y_diff)) + \
                1/(2.0*self.lmbd)*trace(Y_diff.T.dot(Y_diff))


    def __str__(self): return "proximal gradient descent for (inner loop of) GLRM"



class AltMin(Algorithm):
    """ Alternating minimization. """
    
    def __init__(self, objY, objX, algY, algX, converge = None):
        """ Alternate between minimizing obj1 and obj2 until convergence. """
        super(AltMin, self).__init__()
        self.objY = objY # list of (subgrad, obj) tuples; X fixed, minimize over y_i's
        self.objX = objX # singleton (subgrad, obj) tuple; y_i's fixed, minimize over X
        self.algY = algY # algorithm for minimizing subproblem Y (objY) (e.g., SGD)
        self.algX = algX # "" "" 
        if converge: self.converge = converge # specify convergence parameters

    def __str__(self): return "alternating minimization for (outer loop of) GLRM"

    def _update(self, X, Y):
        """
        @X, @Y: two variables to minimize over (iteratively)
        return: updated X, Y after one round of
                minimizing over objX and objY using algX, algY
        """
        # XXX this part can be parallelized

        # hold X constant, minimize over y_i's
        for i in range(len(self.objY)): 
            subgrad, obj = self.objY[i]
            
            try:
                shape = subgrad.shape
                if shape > 1: 
                    alpha = max([norm(X[j,:]) for j in range(X.shape[0])])
                    alpha = 0.5/alpha/X.shape[0]
                else: alpha = 0.5/X.shape[1]/X.shape[0]
            except: alpha = 1.0
            
            subpY = self.algY(subgrad, alpha, obj) 
            Y[i] = subpY.minimize([X], Y[i])

        
        # hold y_i's constant, minimize over X 
        subgrad, obj = self.objX
    
        try: 
            if subgrad.shape > 1:
                alpha = 0
                for y_i in Y: alpha = max([norm(y_i[:,j]) for j in range(y_i.shape[1])] + [alpha])
                alpha = 0.5/alpha/sum([y_i.shape[1] for y_i in Y])
            else: alpha = 0.5/sum([y_i.shape[1] for y_i in Y])/X.shape[1]
        except: alpha = 1.0
        
        subpX = self.algX(subgrad, alpha, obj)
        X = subpX.minimize([y_i.T for y_i in Y], X.T).T

        self.converge.val.append((X, Y))
        self.converge.obj.append(subpX.converge.obj[-1])

        return X, Y 
    
