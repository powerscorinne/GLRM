from paths import *
from glrm import GLRM, huber_loss, squared_loss, norm2sq_reg
from numpy.random import randn, seed
from numpy import arange, zeros
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib import gridspec
from numpy.linalg import norm
from random import sample
from itertools import product
seed(1)

"""
Compare the performance of using Huber loss vs. squared loss 
for factor decomposition when noise is asymmetric.
"""

m, n, k = 100, 50, 10
A_true = randn(m, k).dot(randn(k, n))
sym_noise = 0.2*sqrt(k)*randn(m, n)
asym_noise = sqrt(k)*randn(m,n) + 2*abs(sqrt(k)*randn(m,n))

loss_huber = [huber_loss]
loss_pca = [squared_loss]
regX, regsY = norm2sq_reg(0.1), norm2sq_reg(0.1)

rates = arange(0.0,  0.32, 0.02)
NUM_EXP = 10
A_snapshot = []

mse_glrm = zeros((NUM_EXP, len(rates)))
mse_pca = zeros((NUM_EXP, len(rates)))

for i in range(NUM_EXP):
    for j, rate in enumerate(rates):
        corrupted_entries = sample(list(product(range(m), range(n))), int(m*n*rate))
        A = A_true + sym_noise
        for ij in corrupted_entries: A[ij] += asym_noise[ij]
        if i == 0: A_snapshot.append(A)
        As = [A]

        model_huber = GLRM(As, loss_huber, regsY, regX, k)
        model_pca = GLRM(As, loss_pca, regsY, regX, k)

        A_huber = model_huber.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-4,
                outer_max_iters = 100, inner_max_iters = 1000, quiet = True)
        A_pca = model_pca.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-4,
                outer_max_iters = 100, inner_max_iters = 1000, quiet = True)

        mse_glrm[i,j] = norm(A_huber[0] - A_true)/norm(A_true)
        mse_pca[i,j] = norm(A_pca[0] - A_true)/norm(A_true)


## ====================== PLOTS =============================
# figure one (no data scatter plots)
plt.plot(rates, mse_glrm.mean(0))
plt.plot(rates, mse_pca.mean(0), '--')
plt.legend(("glrm", "pca"), loc='upper left')
plt.xlabel("fraction of corrupted entries")
plt.ylabel("relative mse")
plt.xlim([0, rates.max()])
plt.ylim([0, max(mse_glrm.mean(0).max(), mse_pca.mean(0).max())])
plt.title("huber loss with corrupted data (asymmetric noise)", fontsize=12)
plt.savefig('huber_loss_corrupt.eps', bbox_inches = 'tight')
plt.show()

plt.subplot2grid((5,4), (0,0), colspan = 4, rowspan=3)
plt.plot(rates, mse_glrm.mean(0))
plt.plot(rates, mse_pca.mean(0), '--')
plt.legend(("glrm", "pca"), loc = 'upper left', prop={'size':12})
plt.xlabel("fraction of corrupted entries")
plt.ylabel("relative mse")
plt.xlim([0, rates.max()])
plt.ylim([0, max(mse_glrm.mean(0).max(), mse_pca.mean(0).max())])
plt.title("huber loss with corrupted data (asymmetric noise)")

#plt.suptitle("first two dimensions of data", x = .48, y = .30, fontsize=14)
for i in range(4):
    plt.subplot2grid((5,4), (4,i)) # plot first 2 dimensions of n-dimensional data
    plt.xlim([-15, 25])
    plt.ylim([-15, 25])
    plt.tick_params(axis="both", which="both",
            left="off", right = "off", top = "off", bottom = "off",
            labelleft = "off", labelbottom = "off")
    plt.scatter(A_snapshot[int(i*(len(A_snapshot)-1)/3.0)][:,0],
            A_snapshot[int(i*(len(A_snapshot)-1)/3.0)][:,1])
    plt.title("{0:.0f}% corrupted".format(rates[int(i*(len(A_snapshot)-1)/3.0)]*100), fontsize=10)
plt.savefig('huber_loss_corrupt_data.eps', bbox_inches = 'tight')
plt.show()
