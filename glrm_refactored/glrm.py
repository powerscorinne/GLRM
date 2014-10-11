from convergence import Convergence
from numpy import hstack

class GLRM(object):

    def __init__(self, A, loss, regX, regY, k, missing_list = None):
        # XXX assert that lenghts, sizes are correct

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
        self.ch = Convergence() # XXX update convergence

    # XXX def fit
    

