from glrm import GLRM
from functions import hinge_loss, zero_reg, norm1_reg, norm2sq_reg
from numpy.random import randn, seed
from numpy import sign, round, hstack, minimum, maximum, diag
from numpy.linalg import norm, inv, svd
from pretty_plot import visualize_recovery
from time import time
seed(1)

## =================== Problem data ==============================
m, n, k = 50, 50, 10
A = sign(randn(m,k).dot(randn(k,n)))
As = [A]

## ================== Loss functions, regularizers ======================
losses = [hinge_loss]
regsY, regX = norm1_reg(0.1), norm1_reg(0.1) # parameter is lambda_i

# ======================= Model =============================
model = GLRM(As, losses, regsY, regX, k, svd_init=False)
start = time()
At = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-4, outer_max_iters = 
        100, inner_max_iters = 1000, quiet = False)
print time() - start, " seconds"

# ======================= Results ======================
At = At[0]

visualize_recovery(A, At, "original (bool)", n, k)

# compare to pca using encoded labels
start = time()
A_mu = A.mean(0)
A_centered = A - A_mu
A_var = diag(A_centered.T.dot(A_centered))/m
A_std = A_centered/A_var
u, s, v = svd(A_std)
Apca = u[:,:k].dot(diag(s[:k])).dot(v[:k,:])
Apca = sign(Apca*A_var + A_mu)
print time() - start, " seconds for PCA"

# evalulate performance statistics
mce = float((A != At).sum())/(m*n)
mce_pca = float((A != Apca).sum())/(m*n)
print "percent of misclassified boolean data: "
print "GLRM: {0:.2f}% misclassified points".format(100*mce)
print "PCA: {0:.2f}% misclassified points".format(100*mce_pca)


