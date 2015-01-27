from convergence import Convergence
from numpy import repeat, hstack, array, zeros, ones, sqrt, diag, asarray, hstack, vstack, split, cumsum
from numpy.random import randn
from copy import copy
from numpy.linalg import svd
import cvxpy as cp

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

        loss = [L() for L in loss]

        # save necessary info
        self.A, self.k, self.L = A, k, loss
        if converge == None: self.converge = Convergence()
        else: self.converge = converge

        # initialize cvxpy problems
        self._initialize_probs(A, k, missing_list, regX, regY)
       
        
    def factors(self):
        # return X, Y as matrices (not lists of sub matrices)
        return self.X, hstack(self.Y)

    def convergence(self):
        # convergence information for alternating minimization algorithm
        return self.converge

    def predict(self):
        # return decode(XY), low-rank approximation of A
        return hstack([L.decode(Aj, self.X.dot(yj)) for Aj, yj, L in zip(self.A, self.Y, self.L)])

    def fit(self):
        Xv, Yp, pX = self.probX
        Xp, Yv, pY = self.probY
        self.converge.reset()

        # alternating minimization
        while not self.converge.d():
            objX = pX.solve(solver=cp.SCS, use_indirect=True, eps=1e-2)
            Xp.value[:,:-1] = Xv.value

            # can parallelize this
            for ypj, yvj, pyj in zip(Yp, Yv, pY): 
                pyj.solve(solver=cp.SCS, use_indirect=True, eps=1e-2)
                ypj.value = yvj.value 
            
            self.converge.obj.append(objX)

        self._finalize_XY(Xv, Yv)
        return self.X, self.Y

    def _initialize_probs(self, A, k, missing_list, regX, regY):
        
        # useful parameters
        m = A[0].shape[0]
        ns = [a.shape[1] for a in A]
        if missing_list == None: missing_list = [[]]*len(self.L)

        # initialize A, X, Y
        B = self._initialize_A(A, missing_list)
        X0, Y0 = self._initialize_XY(B, k, missing_list)

        # cvxpy problems
        Xv, Yp = cp.Variable(m,k), [cp.Parameter(k+1,ni) for ni in ns]
        Xp, Yv = cp.Parameter(m,k+1), [cp.Variable(k+1,ni) for ni in ns]
        Xp.value = X0
        for yj, yj0 in zip(Yp, Y0): yj.value = yj0

        masks = [ones((m,ni)) for ni in ns]
        for mask, missing in zip(masks, missing_list):
            for (i,j) in missing: mask[i,j] = 0.0

        obj = sum(L(Bj, cp.mul_elemwise(mask, Xv*yj.value[:-1,:] \
                + repeat(yj.value[-1,:], m,0))) + ry(yj)\
                for L, Bj, yj, ry in zip(self.L, B, Yp, regY)) + regX(Xv)
        pX = cp.Problem(cp.Minimize(obj))
        pY = [cp.Problem(cp.Minimize(\
                L(Bj, cp.mul_elemwise(mask, Xp*yj)) + ry(yj))) \
                for L, Bj, yj, ry in zip(self.L, B, Yv, regY)]

        self.probX = (Xv, Yp, pX)
        self.probY = (Xp, Yv, pY)

    def _initialize_A(self, A, missing_list):
        """ Subtract out means of non-missing, standardize by std. """
        m = A[0].shape[0]
        ns = [a.shape[1] for a in A]
        stdev = [zeros(ni) for ni in ns]
        B = []

        # compute stdev for entries that are not missing
        for ni, sv, ai, missing, L in zip(ns, stdev, A, missing_list, self.L):
            
            # collect non-missing terms
            for j in range(ni):
                elems = array([ai[i,j] for i in range(m) if (i,j) not in missing])
                alpha = cp.Variable()
                # calculate standarized energy per column
                sv[j] = cp.Problem(cp.Minimize(\
                        L(elems, alpha*ones(elems.shape)))).solve()/len(elems)
            
            bi = ai/sv # standardize

            # zero-out missing entries (for XY initialization)
            for (i,j) in missing: bi[i,j] = 0
             
            B.append(bi) # save

        self.stdev = stdev
        return B

    def _initialize_XY(self, B, k, missing_list):
        """ Scale by ration of non-missing, SVD, append col of ones, add noise. """
        A = hstack(bi for bi in B)
        m, n = A.shape

        # normalize entries that are missing
        stdev = A.std(0)
        mu = A.mean(0)
        A = (A-mu)/stdev + 0.01*randn(m,n)

        # SVD to get initial point
        u, s, v = svd(A, full_matrices = False)
        u, s, v = u[:,:k], diag(sqrt(s[:k])), v[:k,:]
        X0, Y0 = asarray(u.dot(s)), asarray(s.dot(v))*asarray(stdev)

        # append col of ones to X, row of zeros to Y
        X0 = hstack((X0, ones((m,1)))) + 0.01*randn(m,k+1)
        Y0 = vstack((Y0, mu)) + 0.01*randn(k+1,n)

        # split Y0
        ns = cumsum([bj.shape[1] for bj in B])
        if len(ns) == 1: Y0 = [Y0]
        else: Y0 = split(Y0, ns, 1)
        
        return X0, Y0

    def _finalize_XY(self, Xv, Yv):
        """ Multiply by std. """
        m = Xv.shape.size[0]
        self.X = asarray(hstack((Xv.value, ones((m,1)))))
        self.Y = [asarray(yj.value)*asarray(stdev) for yj, stdev in zip(Yv, self.stdev)]
