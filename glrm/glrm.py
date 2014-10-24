from convergence import Convergence
from algs import SGD, ProxGD, AltMin
from numpy import hstack, ones
from numpy.random import randn
from copy import copy

# XXX does not support splitting over samples yet (only over features to
# accommodate arbitrary losses by column).

class GLRM(object):

    def __init__(self, A, loss, regX, regY, k, missing_list = None, converge = None, algX=SGD, algY=SGD):

        # Turn everything in to lists / convert to correct dimensions
        if not isinstance(A, list): A = [A]
        if not isinstance(loss, list): loss = [loss]
        if not isinstance(regY, list): regY = [regY]
        if len(regY) == 1 and len(regY) < len(loss): 
            regY = [copy(regY[0]) for _ in range(len(loss))]
        if missing_list and not isinstance(missing_list[0], list): missing_list = [missing_list]

        if missing_list == None:
            missingY, missingX = [[]]*len(loss), [[]]*len(loss)
        else:
            missingY = missing_list
            missingX = [[tuple(reversed(a)) for a in b] for b in missing_list]

        # initialize factors randomly
        m = A[0].shape[0]
        self.X = hstack((randn(m,k), ones((m,1))))
        self.Y = [randn(k+1, a_i.shape[1]) for a_i in A]
        
        # save necessary info
        self.A = A
        self.k = k
        self.L = [L(B, missing = m) for B, L, m in zip(A, loss, missingY)]
        self.regX = regX
        self.regY = regY
        self.missing = missingY
        self.supported_algs_inner = [SGD, ProxGD]

        # initialize alternating minimization algorithm
        self.algX, self.algY = algX, algY  #use stochastic gradient descent on inner subprobs
        self.alg = self._init_altmin(A, loss, regX, regY, missingX, missingY, converge)
  

    def _init_altmin(self, A, loss, regX, regY, missingX, missingY, converge):


        # === Helper functions for extracting appropriate information for inner algs ===        # helper for creating inner-loop optimization problems

        def subprobSGD(losses, reg, skip_last_col):
            def subgrad(Xs, Y):
                if not isinstance(Xs, list): Xs = [Xs]
                grad = sum([L.subgrad(X, Y) for L, X in zip(losses, Xs)])
                if skip_last_col: grad[-1:,:] = 0
                grad_r = reg.subgrad(Y)
                grad_r[-1:,:] = 0 # column of 1's is not penalized 
                return grad+grad_r
         
            # used for determining subgradient descent stepsize
            subgrad.shape = max([L.shape for L in loss] + \
                    [r.shape for r in regY] + [regX.shape])       

            def obj(Xs, Y): return sum([L(X, Y) for L, X in zip(losses, Xs)]) + reg(Y)
            return subgrad, obj

        def subprobProxGD(losses, reg, skip_last_col):
            def subgrad(Xs, Y):
                if not isinstance(Xs, list): Xs = [Xs]
                grad = sum([L.subgrad(X, Y) for L, X in zip(losses, Xs)])
                if skip_last_col: grad[-1:,:] = 0
                return grad

            def obj(Xs, Y): return sum([L(X, Y) for L, X in zip(losses, Xs)])
            obj.loss = losses
            obj.reg = reg
            return (subgrad, reg.prox), obj
                
        subprob = {SGD: subprobSGD, ProxGD: subprobProxGD} # see helper fcn below


        # [(subgrad, obj)] where ith entry is used to initialize subproblem 
        # that takes ([X], y_i) as input
        # call subprob helper function above
        subprobsY = [subprob[self.algY]([L(B, missing = m)], r, False) \
                for B, L, m, r in zip(A, loss, missingY, regY)]       
        
        # (subgrad, obj) singleton used to initiate subproblem that takes 
        # ([y_1.T, ... y_l.T], X.T) as input
        subprobX = subprob[self.algX]([L(B, T=True, missing = m) \
                for B, L, m in zip(A, loss, missingX)], regX, True)

        return AltMin(subprobsY, subprobX, self.algY, self.algX, converge)


#     @property
#     def algX(self):
#         for alg in self.supported_algs_inner:
#             if self._algX == alg: return str(alg)
# 
#     @algX.setter
#     def algX(self, alg):
#         if alg == ProxGD or "prox" in alg.lower(): self._algX = ProxGD
#         elif alg == SGD or "grad" in alg.lower(): self._algX = SGD
#         else: 
#             warn("\n '" + alg + "' nor supported.")
#             return
#  

    def factors(self):
        # return X, Y as matrices (not lists of sub matrices)
        return self.X, hstack(self.Y)


    def convergence(self):
        # convergence information for alternating minimization algorithm
        return self.alg.converge


    def predict(self):
        # return decode(XY), low-rank approximation of A
        return hstack([L.decode(self.X.dot(y_i)) for y_i, L in zip(self.Y, self.L)])


    def fit(self):
        # yay! so easy (see AltMin object in algs.py and _init_altmin fcn above)
        self.X, self.Y = self.alg.minimize(self.X, self.Y)



