from convergence import Convergence
from numpy import sqrt, repeat, tile, hstack, array, zeros, ones, sqrt, diag, asarray, hstack, vstack, split, cumsum
from numpy.random import randn
from copy import copy
from numpy.linalg import svd
import cvxpy as cp

# XXX does not support splitting over samples yet (only over features to
# accommodate arbitrary losses by column).

class GLRM(object):

    def __init__(self, A, loss, regX, regY, k, missing_list = None, converge = None, scale=True):
        
        self.scale = scale
        # Turn everything in to lists / convert to correct dimensions
        if not isinstance(A, list): A = [A]
        if not isinstance(loss, list): loss = [loss]
        if not isinstance(regY, list): regY = [regY]
        if len(regY) == 1 and len(regY) < len(loss): 
            regY = [copy(regY[0]) for _ in range(len(loss))]
        if missing_list and not isinstance(missing_list[0], list): missing_list = [missing_list]

        loss = [L(Aj) for Aj, L in zip(A, loss)]

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
        return hstack([L.decode(self.X.dot(yj)) for Aj, yj, L in zip(self.A, self.Y, self.L)])

    def fit(self, max_iters=100, eps=1e-2, use_indirect=False, warm_start=False):
        
        Xv, Yp, pX = self.probX
        Xp, Yv, pY = self.probY
        self.converge.reset()

        # alternating minimization
        while not self.converge.d():
            objX = pX.solve(solver=cp.SCS, eps=eps, max_iters=max_iters,
                    use_indirect=use_indirect, warm_start=warm_start)
            Xp.value[:,:-1] = copy(Xv.value)

            # can parallelize this
            for ypj, yvj, pyj in zip(Yp, Yv, pY): 
                objY = pyj.solve(solver=cp.SCS, eps=eps, max_iters=max_iters,
                        use_indirect=use_indirect, warm_start=warm_start)
                ypj.value = copy(yvj.value)
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
        self.X0, self.Y0 = X0, Y0

        # cvxpy problems
        Xv, Yp = cp.Variable(m,k), [cp.Parameter(k+1,ni) for ni in ns]
        Xp, Yv = cp.Parameter(m,k+1), [cp.Variable(k+1,ni) for ni in ns]
        Xp.value = copy(X0)
        for yj, yj0 in zip(Yp, Y0): yj.value = copy(yj0)
        onesM = cp.Constant(ones((m,1)))

        obj = sum(L(Aj, cp.mul_elemwise(mask, Xv*yj[:-1,:] \
                + onesM*yj[-1:,:]) + offset) + ry(yj[:-1,:])\
                for L, Aj, yj, mask, offset, ry in \
                zip(self.L, A, Yp, self.masks, self.offsets, regY)) + regX(Xv)
        pX = cp.Problem(cp.Minimize(obj))
        pY = [cp.Problem(cp.Minimize(\
                L(Aj, cp.mul_elemwise(mask, Xp*yj) + offset) \
                + ry(yj[:-1,:]) + regX(Xp))) \
                for L, Aj, yj, mask, offset, ry in zip(self.L, A, Yv, self.masks, self.offsets, regY)]

        self.probX = (Xv, Yp, pX)
        self.probY = (Xp, Yv, pY)

    def _initialize_A(self, A, missing_list):
        """ Subtract out means of non-missing, standardize by std. """
        m = A[0].shape[0]
        ns = [a.shape[1] for a in A]
        mean, stdev = [zeros(ni) for ni in ns], [zeros(ni) for ni in ns]
        B, masks, offsets = [], [], []

        # compute stdev for entries that are not missing
        for ni, sv, mu, ai, missing, L in zip(ns, stdev, mean, A, missing_list, self.L):
            
            # collect non-missing terms
            for j in range(ni):
                elems = array([ai[i,j] for i in range(m) if (i,j) not in missing])
                alpha = cp.Variable()
                # calculate standarized energy per column
                sv[j] = cp.Problem(cp.Minimize(\
                        L(elems, alpha*ones(elems.shape)))).solve()/len(elems)
                mu[j] = alpha.value

            offset, mask = tile(mu, (m,1)), tile(sv, (m,1))
            mask[mask == 0] = 1
            bi = (ai-offset)/mask # standardize

            # zero-out missing entries (for XY initialization)
            for (i,j) in missing: bi[i,j], mask[i,j] = 0, 0
             
            B.append(bi) # save
            masks.append(mask)
            offsets.append(offset)
        self.masks = masks
        self.offsets = offsets
        return B

    def _initialize_XY(self, B, k, missing_list):
        """ Scale by ration of non-missing, SVD, append col of ones, add noise. """
        A = hstack(bi for bi in B)
        m, n = A.shape

        # normalize entries that are missing
        if self.scale: stdev = A.std(0)
        else: stdev = ones(n)
        mu = A.mean(0)
        C = sqrt(1e-2/k) # XXX may need to be adjusted for larger problems
        A = (A-mu)/stdev + C*randn(m,n)

        # SVD to get initial point
        u, s, v = svd(A, full_matrices = False)
        u, s, v = u[:,:k], diag(sqrt(s[:k])), v[:k,:]
        X0, Y0 = asarray(u.dot(s)), asarray(s.dot(v))*asarray(stdev)

        # append col of ones to X, row of zeros to Y
        X0 = hstack((X0, ones((m,1)))) + C*randn(m,k+1)
        Y0 = vstack((Y0, mu)) + C*randn(k+1,n)

        # split Y0
        ns = cumsum([bj.shape[1] for bj in B])
        if len(ns) == 1: Y0 = [Y0]
        else: Y0 = split(Y0, ns, 1)
        
        return X0, Y0

    def _finalize_XY(self, Xv, Yv):
        """ Multiply by std, offset by mean """
        m, k = Xv.shape.size
        self.X = asarray(hstack((Xv.value, ones((m,1)))))
        self.Y = [asarray(yj.value)*tile(mask[0,:],(k+1,1)) \
                for yj, mask in zip(Yv, self.masks)]
        for offset, Y in zip(self.offsets, self.Y): Y[-1,:] += offset[0,:]
            
