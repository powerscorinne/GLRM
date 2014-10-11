from numpy import ones, round, zeros, expand_dims, Inf, tile, arange, repeat, array
from functools import wraps

def scale(A, missing, T):
    """
    Function wrapper for scaling losses
    and filtering out entries marked as missing.
    """

    m, n = A.shape

    # zeros out missing entries
    mask0 = ones(A.shape)
    for indx in missing: mask0[indx] = 0

    def generate_scaled_fcn(fcn):

        if not T: # minimization over Y; columns must be scaled

            # col means of zero-input
            X0, Y0 = zeros((m, 1)), zeros((1,n))
            zero_out = fcn(X0, Y0)*mask0
            zero_col = zero_out.mean(0, keepdims=True)/mask0.mean(0)
            
            # col means of mean-input
            mean_out = fcn(ones((m,1)), zero_col)*mask0
            mean_col = mean_out.mean(0)/mask0.mean(0) # error using col mean
            for i, e in enumerate(mean_col): # handle pathological case
                if e == 0: mean_col[i] = 1

            # scale missing-entry filter
            if mean_out.mean() == 0: scale = 1 # don't want to zero the mask out!!
            else: scale = mean_out.mean()
            mask = mask0/mean_col*scale
        
        else: mask = mask0 # do not scale for minimization over X

        # call function with missing/scaled filter
        @wraps(fcn) # maintain fcn's docstrings
        def scaled_fcn(X, Y):
            return (fcn(X, Y)*mask).sum()
        scaled_fcn.mask = mask

        return scaled_fcn
    return generate_scaled_fcn
