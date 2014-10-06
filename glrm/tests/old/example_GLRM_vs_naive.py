from glrm import GLRM
from functions import squared_loss, hinge_loss, ordinal_loss_flat, zero_reg, \
    norm1_reg, norm2sq_reg
from numpy.random import randn, seed, choice
from numpy import sign, round, hstack, minimum, maximum, diag, zeros, arange, savez
from numpy.linalg import norm, inv, svd
from time import time
from itertools import product
from matplotlib import pyplot as plt
from numpy import log, ones
seed(1)
""" 

Compare GLRM to PCA on data encoded into the reals as
     - k is varied (k_true remains the same) (heterogenous dataset)
     - number of categories of ordinal data increases (homogeneous dataset)

"""


## =================== Problem data ==============================
# Generate heterogenous data
m, n1, n2, n3, k, = 100, 40, 30, 30, 10 # medium hetergenous problem
A = randn(m,k).dot(randn(k,n1+n2+n3)) # everything rank k
A = A + k/20.0*randn(m,n1 + n2 + n3)
A1 = A[:,:n1] # numerical
A2 = sign(A[:,n1:(n1+n2)]) # boolean
A3 = A[:,(n1+n2):]
A3 = A3 - A3.min()
A3 = maximum(minimum(A3/A3.mean()*6 - 1, 6), 0)
A3 = abs(round(A3)) # ordinal
A_true = hstack((A for A in [A1, A2, A3]))

# Missing data
missing_rates = [0.1, 0.1, 0.1]

missing = []
# for each type of loss
for A, n, rate, offset in zip([A1, A2, A3], [n1, n2, n3], missing_rates, [2.0,
    0.5, 0.5]):
    
    missingA = []
    # for each column
    for j in range(n):
        p = abs(A[:,j] + offset)
        #p = ones(len(A[:,j])) # uniform
        p = p/p.sum()

        indx = choice(range(m), int(rate*m), replace = False, p = p)
        for i in indx:
            missingA.append((i,j))
            A[i,j] = 0
        
        Ameans = A[:,j].mean()*(m/(m-len(indx)))
        for i in indx: A[i,j] = Ameans
    
    missing.append(missingA)


# 
#     # probability of drawing each entry of A (weighted by value)
#     p = abs(A + offset).flatten()
#     p = p/p.sum()
#     
#     # choose indices to be removed, encoded as a value in 0, ..., mn-1
#     indx = choice(range(len(p)), int(rate*m*n), replace = False, p = p)
#     
#     # decode indices to be tuples (i,j), which is stored in Amissing, 
#     # and replace A[i,j] with 0
#     lookup = arange(m*n).reshape(m,n)
#     missingA = []
#     nmiscol = zeros(n) # number missing per column
#     for ij in product(range(m), range(n)):
#         if lookup[ij] in indx:
#             missingA.append(ij)
#             A[ij] = 0
#             nmiscol[ij[1]] += 1
# 
#     # with all missing entries removed, replace all missing values with column means
#     Ameans = A.mean(0)*(m/(m-nmiscol))
#     for ij in missingA: A[ij] = Ameans[ij[1]]
# 
#     # save list of missing indices
#     missing.append(missingA)
    
# store A's
As = [A1, A2, A3]

## ================== Loss functions, regularizers ======================
# separate into blocks (associated with different data types / losses)
losses_glrm = [squared_loss, hinge_loss, ordinal_loss_flat]
losses_pca = [squared_loss, squared_loss, squared_loss]
regsY, regX = norm2sq_reg(0.1), norm2sq_reg(0.1) # parameter is lambda_i

k_range = range(int(k/2.5), int(k*1.5)) # range of k's to try 
pca_reals_error, pca_bool_error, pca_ordinal_error =  [], [], []
glrm_reals_error, glrm_bool_error, glrm_ordinal_error = [], [], []

for k in k_range:
    # ======================= GLRM =============================
    # Model
    model = GLRM(As, losses_glrm, regsY, regX, k, missing, svd_init=False)
    At = model.alt_min(outer_RELTOL = 1e-4, inner_RELTOL = 1e-5, outer_max_iters = 
            100, inner_max_iters = 1000, quiet = True)

    # Results
    Am = hstack((A for A in As)) 
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
    At = model.alt_min(outer_RELTOL = 1e-4, inner_RELTOL = 1e-5, outer_max_iters = 
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
plt.ylim((0.0, 0.7))
plt.legend(('glrm', 'pca'))
plt.title('numerical data')

plt.subplot(3,1,2)
plt.plot(k_range, glrm_bool_error)
plt.plot(k_range, pca_bool_error, '--')
plt.ylim((0.0, 0.7))
plt.legend(('glrm', 'pca'))
plt.title('boolean data')

plt.subplot(3,1,3)
plt.plot(k_range, glrm_ordinal_error)
plt.plot(k_range, pca_ordinal_error, '--')
plt.ylim((0.0, 0.7))
plt.legend(('glrm', 'pca'))
plt.title('ordinal data (7 categories)')

plt.show()

#savez('data_GLRM_pca_missing', glrm_missing 
