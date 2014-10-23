from numpy import ones, round, zeros, expand_dims, Inf, tile, arange, repeat, array
from functools import wraps
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from numpy.ma import masked_where

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

def pplot(As, titles):
    # setup
    try: vmin = min([A.min() for A, t in zip(As[:-1], titles) if "missing" not in t]) # for pixel color reference
    except: vmin = As[0].min()
    try: vmax = max([A.max() for A, t in zip(As[:-1], titles) if "missing" not in t])
    except: vmax = As[0].max()
    my_dpi = 96
    plt.figure(figsize=(1.4*(250*len(As))/my_dpi, 250/my_dpi), dpi = my_dpi)
    for i, (A, title) in enumerate(zip(As, titles)):
        plt.subplot(1, len(As), i+1)
        if i == len(As)-1: vmin, vmax = A.min(), A.max()
        if "missing" in title:
            missing = A
            masked_data = ones(As[0].shape)
            for i,j in missing:  masked_data[i,j] = 0
            masked_data = masked_where(masked_data > 0.5, masked_data)
            plt.imshow(As[0], interpolation = 'nearest', vmin = vmin, vmax = vmax)
            plt.colorbar()
            plt.imshow(masked_data, cmap = cm.binary, interpolation = "nearest")
        else:
            plt.imshow(A, interpolation = 'nearest', vmin = vmin, vmax = vmax)
            plt.colorbar()
        plt.title(title)
        plt.axis("off")
   
    plt.show()

def unroll_missing(missing, ns):
    missing_unrolled = []
    for i, (MM, n) in enumerate(zip(missing, ns)):
        for m in MM:
            n2 = m[1] + sum([ns[j] for j in range(i)])
            missing_unrolled.append((m[0], n2))
    return missing_unrolled


