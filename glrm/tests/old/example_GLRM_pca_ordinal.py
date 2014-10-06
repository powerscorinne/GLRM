from glrm import GLRM
from functions import squared_loss, hinge_loss, ordinal_loss_flat, zero_reg, norm1_reg
from numpy.random import randn, seed
from numpy import sign, round, hstack, minimum, maximum, diag, zeros
from numpy.linalg import norm, inv, svd
from time import time
from random import sample
from itertools import product
from matplotlib import pyplot as plt
seed(1)
""" 

Compare GLRM to PCA on data encoded into the reals as
     - k is varied (k_true remains the same) (heterogenous dataset)
     - number of categories of ordinal data increases (homogeneous dataset)

"""


## =================== Problem data ==============================
m, n, k, = 100, 100, 10 # medium hetergenous problem
A = randn(m,k).dot(randn(k,n)) # everything rank k
A = A + 0.5*randn(m,n)
M_range = range(2, 16)

## ================== Loss functions, regularizers ======================
losses_glrm = [ordinal_loss_flat]
losses_pca = [squared_loss]
regsY, regX = norm2sq_reg(0.1), norm2sq_reg(0.1)


for M in M_range:
    AM = A - A.min()
    AM = maximum(minimum(AM/AM.mean()*M/2.0 - 1, M), 0)
    AM = abs(round(AM)) # ordinal
    As = [AM]

    pca_M_error, glrm_M_error = [], []

    # Model
    model = GLRM(As, losses_glrm, regsY, regX, k, missing, svd_init=False)
    At = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-4, outer_max_iters = 
            100, inner_max_iters = 1000, quiet = True)

    # Results
    Am = 
    Atm = hstack((A for A in At))

    mse = norm(As[0] - At[0])/norm(As[0])
    mce = float((As[1] != At[1]).sum())/(m*n2)
    pm = float((As[2] != At[2]).sum())/(m*n3)
    glrm_reals_error.append(mse)
    glrm_bool_error.append(mce)
    glrm_ordinal_error.append(pm)
    
    # ====================== PCA ========================
    # Model
    model = GLRM(As, losses_pca, regsY, regX, k, svd_init=False)
    At = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-4, outer_max_iters = 
            100, inner_max_iters = 1000, quiet = True)

    # Results
    Apca_enc = hstack((A for A in At))
    Apca = zeros(Apca_enc.shape)
    Apca[:, :n1] = Apca_enc[:, :n1]
    Apca[:,n1:n1+n2] = sign(Apca_enc[:,n1:n1+n2]) # decode back into boolean
    Apca[:,n1+n2:] = minimum(maximum(round(Apca_enc[:,n1+n2:]), 0), 6)

    mse_pca = norm(As[0] - Apca[:,:n1])/norm(As[0])
    mce_pca = float((As[1] != Apca[:,n1:n1+n2]).sum())/(m*n2)
    pm_pca = float((As[2] != Apca[:,n1+n2:]).sum())/(m*n3)
    pca_reals_error.append(mse_pca)
    pca_bool_error.append(mce_pca)
    pca_ordinal_error.append(pm_pca)

plt.subplot(3,1,1)
plt.plot(k_range, glrm_reals_error)
plt.plot(k_range, pca_reals_error, '--')
plt.legend(('glrm', 'pca'))
plt.title('numerical data')

plt.subplot(3,1,2)
plt.plot(k_range, glrm_bool_error)
plt.plot(k_range, pca_bool_error, '--')
plt.ylim((0.0, 0.6))
plt.legend(('glrm', 'pca'))
plt.title('boolean data')

plt.subplot(3,1,3)
plt.plot(k_range, glrm_ordinal_error)
plt.plot(k_range, pca_ordinal_error, '--')
plt.ylim((0.0, 0.6))
plt.legend(('glrm', 'pca'))
plt.title('ordinal data (7 categories)')

plt.show()


