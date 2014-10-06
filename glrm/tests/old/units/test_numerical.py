from glrm import GLRM
from functions import squared_loss, zero_reg, norm1_reg
from numpy.random import randn, seed
from numpy import hstack, minimum, maximum
from numpy.linalg import norm
from pretty_plot import visualize_recovery
from time import time
seed(1)

## =================== Problem data ==============================
m, n, k, = 100, 50, 10
A = randn(m,k).dot(randn(k,n)) # everything rank k
As = [A]

## ================== Loss functions, regularizers ======================
losses = [squared_loss]
regsY, regX = norm1_reg(0.1), norm1_reg(0.1)

# ======================= Model =============================
model = GLRM(As, losses, regsY, regX, k, svd_init=False)
start = time()
At = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-4, outer_max_iters = 
        1000, inner_max_iters = 100, quiet = False)
print time() - start, " seconds"

# ======================= Results ======================
At = At[0]
X, Y = model.X, model.Y[0]

visualize_recovery(A, At, "original (numerical)", n, k)

# evalulate performance statistics
mse = norm(A - At)/norm(A)
print "relative error of numerical data: ", mse
