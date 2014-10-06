from glrm import GLRM
from functions import squared_loss, hinge_loss, ordinal_loss, zero_reg, \
        norm1_reg, norm2sq_reg
from numpy.random import randn, seed
from numpy import sign, round, hstack, minimum, maximum, arange, zeros, savetxt, mean
from numpy.linalg import norm, inv
from itertools import product
from random import sample
from pretty_plot import visualize_recovery 

"""

Specify how much of each data block is missing (missing_rates) and see how well
the missing entries are recovered.

"""

seed(1)

## ================= Problem data ==========================
m, n1, n2, n3, k = 200, 100, 50, 50, 15
A = randn(m,k).dot(randn(k,n1+n2+n3))
A1 = A[:,:n1] # numeric data
A2 = sign(A[:,n1:n1+n2]) # boolean data
A3 = A[:,n1+n2:]
A3 = A3 - A3.min()
A3 = maximum(minimum(A3/A3.mean()*5 - 1, 7), 0)
A3 = abs(round(A3)) # ordinal data
As = [A1, A2, A3]

## ================= Missing data ==================
# fraction missing data for each block
missing_rates = [0.1, 0.7, 0.1] # most boolean values are missing
missing = [sample(list(product(range(m), range(n))), int(m*n*rate)) for n, rate
        in zip([n1, n2, n3], missing_rates)]

## ============== Loss functions, regularizers ==================
losses = [squared_loss, hinge_loss, ordinal_loss]
regsY, regX = norm2sq_reg(nu = 0.1), norm2sq_reg(0.1)

## ============== Model =================================
model = GLRM(As, losses, regsY, regX, k, missing)
At = model.alt_min(outer_RELTOL = 1e-3, inner_RELTOL = 1e-5, outer_max_iters =
        100, inner_max_iters = 100, quiet = True)

## ================= Results ===========================
Am = hstack((A for A in As)) # matrix version of As
Atm = hstack((A for A in At)) # matrix version of At

#visualize_recovery(Am, Atm, "original (with missing)", n1, k)

# evaluate performance regarding 'missing data'
mse = norm([As[0][indx] - At[0][indx] for indx in missing[0]])/norm([As[0][indx]
    for indx in missing[0]])
mce = float(sum([As[1][indx] != At[1][indx] for indx in
    missing[1]]))/len(missing[1])
pm = float(sum([As[2][indx] != At[2][indx] for indx in
    missing[2]]))/len(missing[2])
ae = mean([abs(As[2][indx] - At[2][indx]) for indx in missing[2]])/pm

# evaluate performance for all values
mse_all = norm(As[0] - At[0])/norm(As[0])
mce_all = float((As[1] != At[1]).sum())/(m*n2)
pm_all = float((As[2] != At[2]).sum())/(m*n3)
ae_all = abs(As[2] - At[2]).mean()/pm_all

print "error over missing data (error over entire data set)"
print "relative error of numerical data: {0:.4f} ({1:.4f})".format(mse, mse_all)
print "percent misclassified of boolean data: {0:.2f}% ({1:.2f}%)".format(mce*100, mce_all*100)
print "percent mislabeled of ordinal data: {0:.2f}%, ({1:.2f}%)".format(100*pm, 100*pm_all)
print "average error of mislabeled ordinal data: {0:.4f}, ({1:.4f})".format(ae, ae_all)

