from paths import *
from glrm import GLRM, squared_loss, hinge_loss, ordinal_loss, zero_reg, \
        norm1_reg, norm2sq_reg
from numpy.random import randn, seed
from numpy import sign, round, hstack, minimum, maximum, arange, zeros, savetxt, mean
from numpy.linalg import norm, inv
from itertools import product
from random import sample
from glrm.utils.pretty_plot import visualize_recovery_missing

seed(1)

## ================= Problem data ==========================
m, n1, n2, n3, k = 100, 40, 30, 30, 10
A = randn(m,k).dot(randn(k,n1+n2+n3))
A1 = A[:,:n1] # numeric data
A2 = sign(A[:,n1:n1+n2]) # boolean data
A3 = A[:,n1+n2:]
A3 = A3 - A3.min()
A3 = maximum(minimum(A3/A3.mean()*5 - 1, 7), 0)
A3 = abs(round(A3)) # ordinal data
As = [A1, A2, A3]
A_true = hstack((A for A in As))

missing = [list(product(range(30, 80), range(37, 40))),
        list(product(range(30, 80), range(0, 30))),
        list(product(range(30, 80), range(0, 30)))]

for A, ms in zip(As, missing):
    for indx in ms:
        A[indx] = 0
    colmeans = A.mean(0)
    for indx in ms:
        A[indx] = colmeans[indx[1]]
missingm = missing[0] + [(ij[0], ij[1] + n1) for ij in missing[1]] + [(ij[0], ij[1] + n1 + n2) for ij in missing[2]]

## ====================== GLRM =============================
losses = [squared_loss, hinge_loss, ordinal_loss]
regsY, regX = norm2sq_reg(nu = 0.1), norm2sq_reg(0.1)

model = GLRM(As, losses, regsY, regX, k, missing)
At = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-5, outer_max_iters =
        100, inner_max_iters = 100, quiet = True)

Am = hstack((A for A in As)) # matrix version of As
Atm = hstack((A for A in At)) # matrix version of At

visualize_recovery_missing(A_true, Atm, missingm, "mixed data types", "glrm rank {0} recovery".format(k), 
        n1, k, 3, "missing_glrm_block")

## ======================== regularized PCA ==========================
losses = [squared_loss, squared_loss, squared_loss]

model = GLRM(As, losses, regsY, regX, k)
At = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-5, outer_max_iters =
        100, inner_max_iters = 100, quiet = True)

Atm = hstack((At[0], sign(At[1]), round(maximum(minimum(At[2], 6), 0))))

visualize_recovery_missing(A_true, Atm, missingm, "mixed data types", "pca rank {0} recovery".format(k),
        n1, k, 3, "missing_pca_block")


