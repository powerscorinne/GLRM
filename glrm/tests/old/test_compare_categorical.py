from glrm import GLRM
from functions import squared_loss, hinge_loss, ordinal_loss, categorical_loss, \
        zero_reg, norm1_reg
from numpy.random import randn, seed, rand
from numpy import sign, round, hstack, minimum, maximum, zeros, argmax
from numpy.linalg import norm, inv
from matplotlib import pyplot as plt
from time import time
from math import floor

""" 

Test the functionality of using a 'lifted' (multidimensional) loss function.

 - It turns out that lifting does not do much for Boolean data
 - May return to this experiment with other lifted data types

"""

#seed(2)

## =================== Problem data ==============================
m, n = 50, 20
ks = range(5,6)
NUM_EXP = 1
M = 4
A = round((M-1)*rand(m,n)) # categorical
Ab = zeros((A.shape[0], M*A.shape[1])) # encode as boolean
for i in range(m):
    for j in range(n):
        Ab[i,j*M+A[i,j]] = 1


## ================== Loss functions, regularizers ======================
A_lifted = [A]
A_bool = [A_b]
losses_lifted = [categorical_loss]
losses_bool = [hinge_loss]
regsY, regX = norm1_reg(0.1), norm1_reg(0.1) # parameter is lambda_i

# ======================= Models =============================
ce_lifted, ce_bool = zeros((NUM_EXP, len(ks))), zeros((NUM_EXP, len(ks)))

start = time()
for i in range(NUM_EXP):
    for j in range(len(ks)):
        k = ks[j]

        # vector encoding
        model_lifted = GLRM(A_lifted, losses_lifted, regsY, regX, k)
        A_lifted_out = model_lifted.alt_min(outer_RELTOL = 1e-2, inner_RELTOL = 1e-5, 
                outer_max_iters = 100, inner_max_iters = 100, quiet=False)
        ce_lifted[i,j] = float((A != A_lifted_out[0]).sum())/(m*n)

        # scalar encoding
        model_bool = GLRM(A_bool, losses_bool, regsY, regX, k)
        #seed(1)
        A_bool_out = model_bool.alt_min(outer_RELTOL = 1e-2, inner_RELTOL = 1e-5, 
                outer_max_iters = 100, inner_max_iters = 100)
        A_out = zeros(A.shape)
#        for i in range(m):
#            for j in range(n):
#                A_out[i, j] = argmax([A_bool_out[i,M*j:M*(j+1)]])
#        ce_bool[i,j] = float((A_b != A_out).sum())/(m*n)

    # progress
    ti = time() - start
    print "experiment {0}/{1} in {2:02d}:{3:02d}".format(i+1, NUM_EXP,
            int(floor(ti/60)), int(ti%60))

cel = ce_lifted.mean(0)
#ceb = ce_bool.mean(0)
# 
# def visualize(cel, ces, ks):
#     plt.plot(ks, cel, '-')
#     plt.plot(ks, ces, '--')
#     plt.legend(("vector encoding", "scalar encoding"))
#     plt.xlabel("k")
#     plt.ylabel("misclassification rate")
#     plt.title("Boolean encoding: random {0} x {1} table".format(m, n))
#     plt.show()
# 
# visualize(cel, ces, ks)
# 
