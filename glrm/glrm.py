from convergence import Convergence
from numpy import hstack, ones
from numpy.random import randn
from copy import copy

# XXX does not support splitting over samples yet (only over features to
# accommodate arbitrary losses by column).

class GLRM(object):

    def __init__(self, A, loss, regX, regY, k, missing_list = None, converge = None):

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
        
        # useful parameters
        m = A[0].shape[0]
        ns = [a.shape[1] for a in A]

        # XXX need to add offest

        # create cvxpy problems
        Xp = cp.Parameter(m,k+1)
        Y = [cp.Variable(k+1, ni) for ni in ns]
        LossY = [L(B, missing=m) for L, B, m in zip(loss, A, missingY)]
        objY = [cp.Problem(L(Xp, yj) + r(yj)) for L, yj, r in zip(LossY, Y, regY)]
        self.probsY = (Xp, Y, [cp.Problem(objyj) for objyj in objY])

        X = cp.Variable(m,k)
        Yp = [cp.Parameter(k+1, ni) for ni in ns]
        LossX = [L(B, T=True, missing=m) for L, B, m in zip(loss, A, missingX)]
        objX = cp.Problem(sum(L(X, yj) for yj in Yp) + regX(X))
        self.probX = (X, Yp, cp.Problem(objX))
               
        # save necessary info
        self.A = A
        self.k = k
        self.L = [L(B, missing = m) for B, L, m in zip(A, loss, missingY)]
        self.missing = missingY
        if converge == None: self.converge = Convergence()
        else: self.converge = converge

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
        Xp, Y, pY = self.probsY
        X, Yp, pX = self.probX
        
        # initialize X randomly
        Xp.value = hstack((randn(m,k), ones((m,1))))
        self.converge.reset()

        # iterate until converged
        while not self.converge.d():
            for i, (yj, pyj) in enumerate(zip(Y, pY)): 
                pyj.solve()
                Yp[i].value = pyj.value

            obj = pX.solve()
            Xp.value = X.value

            self.converge.obj.append(obj)
        
        return X.value, [yj.value for yj in Y]





