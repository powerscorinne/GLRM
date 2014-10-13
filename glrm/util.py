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

def pretty_plot(A, A_hat, missing):
    # setup
    vmin = min(A.min(), A_hat.min()) # for pixel color reference
    vmax = max(A.max(), A_hat.max())
    my_dpi = 96
    plt.figure(figsize=(1400/my_dpi, 250/my_dpi), dpi = my_dpi)
    if missing == [[]]

    plt.subplot(1, 4, 1)
    plt.imshow(A, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title("original")
    plt.tick_params(\
            axis = "both",
            which = "both",
            left = "off",
            right = "off",
            top = "off",
            labelleft = "off",
            labelbottom = "off")

    plt.subplot(1,4,2)
    masked_data = ones(A.shape)
    for ij in missing: masked_data[ij] = 0
    masked_data = masked_where(masked_data > 0.5, masked_data)
    plt.imshow(A, interpolation = "nearest", vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.imshow(masked_data, cmap = cm.binary, interpolation = "nearest")
    plt.title("entries removed")
    plt.tick_params(\
            axis = "both",
            which = "both",
            left = "off",
            right = "off",
            top = "off",
            labelleft = "off",
            labelbottom = "off")

    plt.subplot(1,4,3)
    plt.imshow(A_hat, interpolation = "nearest", vmin = vmin, vmax = vmax)
    plt.tick_params(\
            axis = "both",
            which = "both",
            left = "off",
            right = "off",
            top = "off",
            labelleft = "off",
            labelbottom = "off")
    plt.title("low rank approx")
    plt.colorbar()

    plt.subplot(1,4,4)
    B = A - A_hat
    plt.imshow(B, interpolation = "nearest", vmin = B.min(), vmax = B.max())
    plt.colorbar()
    plt.tick_params(\
            axis = "both",
            which = "both",
            left = "off",
            right = "off",
            top = "off",
            labelleft = "off",
            labelbottom = "off")
    plt.title("error")
    
    plt.show()

