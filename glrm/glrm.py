from numpy.random import randn
from numpy.linalg import svd
from numpy import Inf, hstack, ones, diag, split, vstack, zeros, cumsum
from warnings import warn
from matplotlib import pyplot as plt
from block import Block
from utils import do_cprofile

"""
GLRM object is initialized with data A (as a list, separated by feature date
type) and losses (also a list, of generic loss functions (e.g. squared_loss)
corresponding to As). 

This object handles the creation of the appropriate blocks, and the alternating
minimization step. Also, there is where I've stored the visualization tool.
"""

# XXX there is plenty of housekeeping that needs to be done here...

class GLRM(object):

    def __init__(self, As, losses, regsY, regX, k, missings = None,
            svd_init=False):
        # XXX assert that lengths, sizes of input are correct

        # As, losses, and regsY can (and should be) lists
        if not isinstance(As, list): As = [As]
        if not isinstance(losses, list): losses = [losses]
        # in case one blanket regY in given when Y has multiple features:
        if not isinstance(regsY, list): regsY = [regsY]*len(As)
               
        # format missing data for each block
        if missings == None: 
            missingsY = [[]]*len(losses)
            missingsX = [[]]*len(losses)
        else:
            missingsY = missings
            missingsX = [[tuple(reversed(a)) for a in b] for b in missings]
        
        # create blocks for Y (feature) updates
        lossesY = [L(A, missing = m) for A, L, m in zip(As, losses, missingsY)]
        self.blocksY = [Block(loss, regY) for loss, regY in zip(lossesY, regsY)]
        # an array of blocks, each block with one loss (as a list)
        # update features separately (as split by As, losses)
        
        # create blocks for X (sample) updates
        lossesX = [L(A, T = True, missing = m) for A, L, m in zip(As, losses, missingsX)]
        self.blocksX = Block(lossesX, regX) # all samples updated in one block 
        # one block (as a list), with an array of losses
        # (no spitting over samples yet)
        
        # save As, X, Y
        self.As = As
        m = As[0].shape[0]
        # XXX probably should move this to alt_dir in block
        # XXX also, is broken
        if svd_init == True: # save initial X, Y as svd factors
            u, s, v = svd(hstack(b.L[0].encode(A) for b, A in zip(self.blocksY, As)), full_matrices = False)
            n = v.shape[1]
            self.X = hstack((u[:,:k].dot(diag(s[:k])), ones((m,1))))
            Y = v[:k,:]
            Y = vstack((Y, zeros((1,Y.shape[1]))))
            cuts = [b.L[0].n for b in self.blocksY]
            cuts = cumsum(cuts[:-1])
            self.Y = split(Y, cuts, 1)
        else:
            self.X = hstack((randn(m,k), ones((m,1)))) # initialize random X (with offset)
            self.Y = [randn(k+1,ni) for ni in [b.L[0].n for b in self.blocksY]] # Y
    
    def __call__(self, X = None, Y = None):
        # includes regularization loss
        if X == None or Y == None:
            X, Y  = self.X, self.Y
        return self.blocksX([Yp.T for Yp in Y], X.T) + self.blocksX.r(X.T) + sum([b.r(Yp) for b, Yp in zip(self.blocksY, Y)])

    #@do_cprofile
    def alt_min(self, outer_RELTOL = 1e-4, inner_RELTOL = 1e-5, outer_max_iters
            = 1000, inner_max_iters = 100, quiet = True):
        # alternating minimization
        f0 = Inf 
        # n*m, in oh so many words...
        size = sum([b.L[0].n for b in self.blocksY])*self.blocksX.L[0].n
        
        for i in range(outer_max_iters):
            # update Y
            for b, Y in zip(self.blocksY, self.Y): # split over features
                b.update(self.X, Y, inner_RELTOL, inner_max_iters)
            
            # update X
            self.blocksX.update([Y.T for Y in self.Y], self.X.T,
                    inner_RELTOL, inner_max_iters, skip_last_col = True) # all samples at once

            # loss of XY
            f = self()
            if abs(f - f0) < outer_RELTOL*size: 
                print i, " iterations"
                break
            if not quiet: print f
            f0 = f
            if i == outer_max_iters-1: warn("hit max iters for alternating minimization")
 
        # decode and return At as list analogous to input As = [A1, A2, ...]
        return [b.L[0].decode(self.X.dot(Y)) for b, Y in zip(self.blocksY,
            self.Y)]
