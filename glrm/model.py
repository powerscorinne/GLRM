from convergence import Convergence
from fit import Block
from numpy import hstack, ones
from numpy.random import randn

# XXX does not support splitting over samples yet (only over features to
# accommodate arbitrary losses by column).

class GLRM(object):

    def __init__(self, A, loss, regX, regY, k, missing_list = None):

        # A, loss, and regY should be lists
        if not isinstance(A, list): A = [A]
        if not isinstance(loss, list): loss = [loss]
        if not isinstance(regY, list): regY = [regY]

        if missing_list == None:
            missingY, missingX = [[]]*len(loss), [[]]*len(loss)
        else:
            missingY = missing_list
            missingX = [[typle(reversed(a)) for a in b] for b in missing_list]

        # create blocks for Y (feature) updates
        lossY = [L(B, missing = m) for B, L, m in zip(A, loss, missingY)]
        self.blocksY = [Block(l, r) for l, r in zip(lossY, regY)]

        # create blocks for X (sample) updates
        lossX = [L(B, T = True, missing = m) for B, L, m in zip(A, loss, missingX)]
        self.blocksX = Block(lossX, regX) # all samples updated as one unit

        # XXX each of these blocks can be updated independently and in parallel !!
        
        # initialize factors randomly
        m = A[0].shape[0]
        self.X = hstack((randn(m,k), ones((m,1))))
        self.Y = [randn(k+1,ni) for ni in [b.L[0].n for b in self.blocksY]]
        self.A = A
        self.converge = Convergence(TOL = 1e-3, max_iters = 1000) # can enter TOL and max_iters

    def factors(self):
        # return X, Y as matrices (not lists of sub matrices)
        return self.X, hstack(self.Y)

    def convergence(self):
        return self.converge

    def predict(self):
        # return decode(XY), low-rank approximation of A
        return hstack([Yf.L[0].decode(self.X.dot(Y)) for Y, Yf in zip(self.Y, self.blocksY)])

    def loss_obj(self):
        # loss functionn evaluated at current (X, Y)
        return self.blocksX([Yp.T for Yp in self.Y], self.X.T)

    def reg_obj(self):
        # regularization cost evaluated at current (X, Y)
        return  self.blocksX.r(self.X.T) + sum([b.r(Yp) for b, Yp in zip(self.blocksY, self.Y)])

    def fit(self):
        # fit using alternating minimization with gradient descent 
        while not self.converge.d():

            # update Y
            convergeY = Convergence(max_iters = 1e3) # again, can sepcify TOL and max_iters
            for b, Y in zip(self.blocksY, self.Y): # split over features
                b.update(self.X, Y, convergeY)

            # update X
            convergeX = Convergence(max_iters = 1e3)
            self.blocksX.update([Y.T for Y in self.Y], self.X.T, \
                    convergeX, skip_last_col = True)

            # update convergence object
            self.converge.val.append(self.factors())
            self.converge.obj.append(self.loss_obj() + self.reg_obj())
