from glrm import GLRM
from functions import categorical_loss, hinge_loss, norm1_reg
from numpy.random import randn, seed, rand
from numpy import sign, round, hstack, minimum, maximum, zeros, diag, argmax
from numpy.linalg import norm, inv, svd
from matplotlib import pyplot as plt
from time import time
from math import floor

""" 

Test the functionality of using a 'lifted' (multidimensional) loss function.
Compare its performance to the scalar case.

"""

## =================== Problem data ==============================
m, n = 100, 50
k = 10
M = 4

#A = round((M-1)*rand(m,n))
A = rand(m,k).dot(rand(k,n))
A = round((A- A.min())/A.max()*(M-1))

## ================== Loss functions, regularizers ======================
As_lifted = [A]
losses_lifted = [categorical_loss]
regsY, regX = norm1_reg(0.1), norm1_reg(0.1) # parameter is lambda_i

# ======================= Models =============================
model_lifted = GLRM(As_lifted, losses_lifted, regsY, regX, k)
A_lifted = model_lifted.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-5, 
        outer_max_iters = 100, inner_max_iters = 100, quiet=False)
print (A_lifted[0] != A).sum()

# 
# # compare ce if just used pca
# u, s, v = svd(Ab, full_matrices=False)
# s[k:] = 0
# A_pca_encoded = u.dot(diag(s)).dot(v)
# A_pca = zeros(A.shape)
# for i in range(m):
#     for j in range(n):
#         A_pca[i,j] = argmax(A_pca_encoded[i,M*j:M*(j+1)])
# ce_pca = float((A_pca != A).sum())/(m*n)
# 
# 
# plt.subplot(1,2,1)
# plt.plot(iters, ce_lifted, '-')
# plt.plot(iters, ce_bool, '--')
# plt.plot(iters, [ce_pca]*len(iters), '-.')
# plt.legend(("lifted", "encoded", "PCA"))
# plt.xlabel("iteration")
# plt.ylabel("misclassification rate")
# plt.title("error")
# frame1 = plt.gca()
# #frame1.axes.get_yaxis().set_ticks([])
#     
# plt.subplot(1,2,2)
# plt.plot(iters, loss_lifted, '-')
# plt.plot(iters, loss_scalar, '--')
# plt.legend(("lifted", "encoded"))
# plt.xlabel("iteration")
# plt.ylabel("objective value")
# plt.title("trajectory")
# frame2 = plt.gca()
# frame2.axes.get_yaxis().set_ticks([])
#     
# plt.show()
