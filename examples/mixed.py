from paths import *
from glrm import GLRM, hinge_loss, zero_reg, norm1_reg, norm2sq_reg, \
        squared_loss, ordinal_loss_lin, categorical_loss
from numpy.random import randn, seed, rand
from numpy import sign, round, hstack, minimum, maximum, diag
from numpy.linalg import norm, inv, svd
from glrm.utils.pretty_plot import visualize_recovery_mixed
from time import time
seed(1)

## =================== Problem data ==============================
m, n1, n2, n3, k = 100, 40, 30, 30, 10
A = randn(m,k).dot(randn(k,n1 + n2 + n3))
A1 = A[:,:n1] # numerical
A2 = sign(A[:,n1:(n1+n2)]) # boolean
A3 = A[:, (n1+n2):]
A3 = A3 - A3.min()
A3 = round(7*rand(m, n3))
A3 = abs(round(A3))
As = [A1, A2, A3]

## ================== GLRM ======================
losses = [squared_loss, hinge_loss, squared_loss]
regsY, regX = norm2sq_reg(0.1), norm2sq_reg(0.1) # parameter is lambda_i
model = GLRM(As, losses, regsY, regX, k, svd_init=False)
At = model.alt_min(outer_RELTOL = 1e-4, inner_RELTOL = 1e-5, outer_max_iters = 
        100, inner_max_iters = 1000, quiet = False)
Am = hstack((A for A in As))
Atm = hstack((A for A in At))

visualize_recovery_mixed(Am, Atm, "mixed data types", "glrm rank {0} recovery".format(k), n1, k, "mixed_data_glrm")

## =================== regularized PCA =====================
losses = [squared_loss, squared_loss, squared_loss]
model = GLRM(As, losses, regsY, regX, k, svd_init=False)
At = model.alt_min(outer_RELTOL = 1e-4, inner_RELTOL = 1e-5, outer_max_iters = 
        100, inner_max_iters = 1000, quiet = False)
Atm = hstack((At[0], sign(At[1]), round(maximum(minimum(At[2], 6), 0))))

visualize_recovery_mixed(Am, Atm, "mixed data types", "pca rank {0} recovery".format(k), n1, k, "mixed_data_pca")
