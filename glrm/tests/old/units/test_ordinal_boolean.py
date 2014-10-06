from glrm import GLRM
from functions import hinge_loss, zero_reg, norm1_reg
from numpy.random import randn, seed
from numpy import sign, round, hstack, minimum, maximum, diag, ones, zeros, argmax
from numpy.linalg import norm, inv, svd
from pretty_plot import visualize_recovery
from time import time
seed(1)

## =================== Problem data ==============================
m, n, k, = 100, 50, 10 # medium hetergenous problem
M = 7
A = randn(m,k).dot(randn(k,n)) # everything rank k
A = A - A.min()
A = maximum(minimum(A/A.mean()*5 - 1, 6), 0)
A = abs(round(A)) # ordinal (likert 1-7)
Ab = -ones((A.shape[0], 7*A.shape[1]))
for i in range(m):
    for j in range(n):
        Ab[i,j*M + A[i,j]] = 1

As = [Ab]

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
A_out = model.X.dot(model.Y[0])
At = zeros(A.shape)
for i in range(m):
    for j in range(n):
        At[i,j] = argmax([A_out[i,7*j:7*(j+1)]])


visualize_recovery(A, At, "original (ordinal encoded as boolean)", n, k)

# compare to pca
# A_mu = Ab.mean(0)
# A_centered = Ab - A_mu
# A_var = diag(A_centered.T.dot(A_centered))/m
# A_std = A_centered/A_var
# u, s, v = svd(A_std)
# Apca_encoded = u[:,:k].dot(diag(s[:k])).dot(v[:k,:])
# Apca = zeros(A.shape)
# for i in range(m):
#     for j in range(n):
#         Apca[i,j] = argmax([Apca_encoded[i,7*j:7*(j+1)]])
 
# evalulate performance statistics
pm = float((A != At).sum())/(m*n)
ae = abs(A - At).mean()/pm

#pm_pca = float((A != Apca).sum())/(m*n)
#ae_pca = abs(A - Apca).mean()/pm_pca

print "percent of mislabeled likert data:"
print "GLRM: {0:.4f}% percent misclassified".format(100*pm)
#print "PCA: {0:.4f}% percent misclassified".format(100*pm_pca)

print "average error of mislabeled likert data:"
print "GLRM: {0:.4f} average error".format(ae)
#print "PCA: {0:.4f} average error".format(ae_pca)

