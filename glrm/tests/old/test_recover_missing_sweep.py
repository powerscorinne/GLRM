from glrm import GLRM
from functions import squared_loss, hinge_loss, ordinal_loss_flat, zero_reg, norm1_reg
from numpy.random import randn, seed
from numpy import sign, round, hstack, minimum, maximum, arange, zeros, savetxt
from numpy.linalg import norm, inv
from itertools import product
from random import sample
from matplotlib import pyplot as plt

"""

Generate data, and attempt to find low rank model with entries withheld.
Compare how well low rank model recovers missing data.

"""

seed(1)

## ================= Problem data ==========================
m, n1, n2, n3, k = 75, 25, 25, 25, 10
A = randn(m,k).dot(randn(k,n1+n2+n3))
A1 = A[:,:n1] # numeric data
A2 = sign(A[:,n1:n1+n2]) # boolean data
A3 = A[:,n1+n2:]
A3 = A3 - A3.min()
A3 = maximum(minimum(A3/A3.mean()*5 - 1, 7), 0)
A3 = abs(round(A3)) # likert data

## ============== Loss functions, regularizers ==================
As = [A1, A2, A3]
losses = [squared_loss, hinge_loss, ordinal_loss_flat]
regsY, regX = norm1_reg(nu = 0.1), norm1_reg(0.1)

## ================= Missing data ==================
rates = arange(0.00, 0.5, 0.02) # fraction missing data (sweep over 0.01 ... 0.6)
NUM_EXP = 10 # number of experiments

exp = "data/recover_missing_"
ext = ".txt"
(errorR, errorB, errorL) = zeros((NUM_EXP, len(rates))), zeros((NUM_EXP, len(rates))), zeros((NUM_EXP, len(rates)))

for i in range(NUM_EXP):
    errorR_missing, errorB_missing, errorL_missing = [], [], []
    for sample_rate in rates:
        print sample_rate

        # generate a list of indices [(i,j), ... ] that are considered missing
        # regardless of the values A[i,j]
        missing = [sample(list(product(range(m), range(n))), int(m*n*sample_rate)) for n in [n1, n2, n3]]
        # to indicate no data are missing, use missing = [] or do not specify in model below

        ## ===================== Model ======================
        model = GLRM(As, losses, regsY, regX, k)

        # A_rec is the 'recovered' low rank representation
        A_rec = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-4,
                outer_max_iters = 100, inner_max_iters = 100)

        ## ==================== Results ======================
        #Am = hstack((A for A in As)) # convert list of matrices to one large matrix
        #Atm = hstack((A for A in A_rec))
        #model.visualize(Am, Atm, "original (real, boolean, likert)", n1, n2, k)

        if sample_rate == 0: errorR_missing.append(0)
        else: errorR_missing.append(norm([As[0][indx] - A_rec[0][indx] for indx in missing[0]])/norm([As[0][indx] for indx in missing[0]]))
        errorB_missing.append(float(sum([As[1][indx] != A_rec[1][indx] for indx in missing[1]]))/(m*n2))
        errorL_missing.append(float(sum([As[2][indx] != A_rec[2][indx] for indx in missing[2]]))/(m*n3))

    errorR[i:i+1,:] = errorR_missing
    errorB[i:i+1,:] = errorB_missing
    errorL[i:i+1,:] = errorL_missing

errorR_missing = errorR.mean(0)
errorB_missing = errorB.mean(0)
errorL_missing = errorL.mean(0)
# this data has been saved to data/recover_missing_*_mean.txt

# plot
plt.subplot(1,2,1)
plt.plot(rates, errorR_missing)
plt.xlabel("fraction removed")
plt.ylabel("relative error")
plt.subplot(1,2,2)
plt.plot(rates, errorB_missing, '-')
plt.plot(rates, errorL_missing, '--')
plt.xlabel("fraction removed")
plt.ylabel("misclassification error")
plt.legend(("boolean", "ordinal"))
plt.show()

