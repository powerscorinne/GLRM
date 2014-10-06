from glrm import GLRM
from functions import categorical_loss, hinge_loss, norm2sq_reg
from numpy.random import randn, seed, rand
from numpy import sign, round, hstack, minimum, maximum, zeros, diag, argmax, \
ones, expand_dims, arange
from numpy.linalg import norm, inv, svd
from matplotlib import pyplot as plt
from time import time
from math import floor

""" 

Test the functionality of using a 'lifted' (multidimensional) loss function.
Compare its performance to the scalar case.

This experiment (n, m, k, NUM_TERS = 100, 50, 5, 50) takes about 2-3 minutes.

"""


## =================== Problem data ==============================

m, n, k = 100, 50, 10
M = 5
NUM_ITERS = 50

iters = range(NUM_ITERS)
seed(1)
#A = round((M-1)*rand(m,k)).astype(int) # full rank category matrix
A = rand(m,k).dot(rand(k,n))
A = round((A - A.min())/A.max()*(M-1)).astype(int)

## ==================== Categorical PCA ===========================

As = [A]
losses = [categorical_loss]
regsY, regX = norm2sq_reg(0.1), norm2sq_reg(0.1)
ce_cat, loss_cat = [], [] # classification error, objective loss

start = time()
for i in iters: # cutoff alt min after i steps 
    seed(1)
    model = GLRM(As, losses, regsY, regX, k, svd_init=True)
    A_cat = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-3, 
            outer_max_iters = i, inner_max_iters = 100, quiet=True)

    ce_cat.append(float((A != A_cat[0]).sum())/(m*n))
    loss_cat.append(model())
print time() - start

# ================ Boolean PCA on encoded categories ==============

# encode A as boolean matrix
Ab = -ones((A.shape[0], M*A.shape[1])) 
x = expand_dims(arange(m),1)
y = (M*arange(n)).repeat(m).reshape(n,m).T + A
Ab[x, y] = 1

As = [Ab]
losses = [hinge_loss] # same regularizers as above
ce_bool, loss_bool = [], []

start = time()
for i in iters:
    seed(1)
    model = GLRM(As, losses, regsY, regX, k, svd_init=True)
    _ = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-3,
            outer_max_iters = i, inner_max_iters = 100, quiet=True)
    
    # extract and decode A_bool from model_bool
    A_bool_encoded = model.X.dot(model.Y[0])
    A_bool = zeros(A.shape)
    for i in range(m):
        for j in range(n):
            A_bool[i,j] = argmax(A_bool_encoded[i,M*j:M*(j+1)])
    ce_bool.append(float((A != A_bool).sum())/(m*n))
    loss_bool.append(model())
print time() - start

# ================== PCA on encoded categories =====================

u, s, v = svd(Ab, full_matrices=False)
s[k:] = 0
A_pca_encoded = u.dot(diag(s)).dot(v)
A_pca = zeros(A.shape)
for i in range(m):
    for j in range(n):
        A_pca[i,j] = argmax(A_pca_encoded[i,M*j:M*(j+1)])
ce_pca = float((A_pca != A).sum())/(m*n)


# =========================== Results ===========================

#print ce_cat, ce_bool, ce_pca 
plt.subplot(1,2,1)
plt.plot(iters, ce_cat, '-')
plt.plot(iters, ce_bool, '--')
plt.plot(iters, [ce_pca]*len(iters), '-.')
plt.legend(("categorical PCA", "boolean PCA", "PCA"))
plt.xlabel("iteration")
plt.ylabel("misclassification rate")
plt.title("error")
frame1 = plt.gca()
#frame1.axes.get_yaxis().set_ticks([])
    
plt.subplot(1,2,2)
plt.plot(iters, loss_cat, '-')
plt.plot(iters, loss_bool, '--')
plt.legend(("categorical PCA", "boolean PCA"))
plt.xlabel("iteration")
plt.ylabel("objective value")
plt.title("trajectory")
frame2 = plt.gca()
frame2.axes.get_yaxis().set_ticks([])
    
plt.show()
