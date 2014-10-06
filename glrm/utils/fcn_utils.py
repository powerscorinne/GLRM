from numpy.random import rand
from numpy import ones, round, zeros, expand_dims, Inf, tile, arange, repeat, array
from functools import wraps
from copy import copy

def min_value(A, missing):
    """ find minimum value in A, ignoring entries flagged by 'missing'. """
    A2 = copy(A)
    try: A2[missing] = Inf
    except: 
        for indx in missing:
            A2[indx] = Inf
    return A2.min()
 
def elementwise_shrinkage(a, kappa):
    """ soft threshold a (scalar, list, or array) with parameter kappa). """
    try: return maximum(a - kappa*ones(a.shape), 0) - maximum(-a -
            kappa*ones(a.shape), 0)
    except: return max(a - kappa, 0) - max(-a - kappa, 0)

def scale(A, T, missing, M = None):
    """ 
    
    Function wrapper for scaling losses, regularizers 
    and filtering out entries marked as missing. 
    
    """
    
    m, n = A.shape

    # zero out missing entries
    mask0 = ones(A.shape)
    for indx in missing: mask0[indx] = 0
    
    def generate_scaled_fcn(fcn):

        if not T: # minimization over Y; columns must be scaled
            
            # col means of zero-input
            if M: # calculate mean of multidimensional functions XXX??
                # XXX use encode function to eliminate this
                maskM = repeat(mask0, M, 1)
                Ae = zeros((m, sum(M))) # XXX or -ones?
                indx = tile(array(M).cumsum() - M, (m,1)) + A
                Ae[expand_dims(arange(m), 1), indx.astype(int)] = 1
                zero_col = Ae.mean(0, keepdims=True)/maskM.mean(0)
            
            else: # calculate mean for scalar functions
                P0, V0 = zeros((m, 1)), zeros((1, n))
                zero_out = fcn(P0, V0)*mask0
                zero_col = zero_out.mean(0, keepdims=True)/mask0.mean(0)

            # col means of mean-input
            mean_out = fcn(ones((m, 1)), zero_col)*mask0
            mean_col = mean_out.mean(0)/mask0.mean(0) # error using col mean
            for i, e in enumerate(mean_col): # handle pathological case
                if e == 0: mean_col[i] = 1
        
            # scale missing-entry filter
            mask = mask0/mean_col
        
        else: mask = mask0 # do not scale for minimization over X
        
        # call function with missing/scaled filter
        @wraps(fcn) # maintain f's doc strings
        def scaled_fcn(P, V):
            return (fcn(P, V)*mask).sum()
        scaled_fcn.mask = mask

        return scaled_fcn
    return generate_scaled_fcn
