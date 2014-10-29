## GLRM

GLRM is a python package for exploratory data analysis using Generalized Low
Rank Models (GLRMs). 

A GLRM seeks factors X and Y such that XY approximates data table A
using an arbitrary error metric (i.e. loss function) for each column of A.
This framework allows for the generalization of principal components analysis
(PCA) to a heterogenous dataset A, where columns of A contain data with
different data types (e.g., Boolean, ordinal, interval). 
GLRM easily handles missing data by choosing a loss of zero for the missing
entries of A.

For more information on GLRMs, see [our
paper](http://www.stanford.edu/~boyd/papers/glrm.html).

This project provides a GLRM object for automatically computing factors X and Y,
decoding XY back into the appropriate domain, and imputing missing entries.

## Installation
python setup.py install

## Basic usage
The source code for similar problems can be found in 'examples' folder.

A GLRM model is specified by data table A, loss functions L, and a list of missing
entries. 

    from glrm import GLRM

Consider a data table A that is approximately rank k, where the first n1 columns
contain Boolean data, and the next n2 columns contain numerical data. 

    m, n1, n2, k = 50, 25, 25, 5
    eta = 0.1 # noise
    A = randn(m,k).dot(randn(k,n1+n2)) + eta*randn(m,n1+n2)
    A_bool = sign(A[:,:n1]) # Boolean data must be labeled as -1, 1
    A_real = A[:,n1:]

We decide to use hinge loss for the Boolean data, and quadratic loss 
for the numerical data. The scaling of each loss function 
is handled automatically during the intialization of the GLRM object. 

    from glrm.loss import QuadraticLoss, HingeLoss

Data A is stored as a list of submatrices, where each submatrix
is associated with a data type. The loss functions associated with each
submatrix are stored similarly.

    A_list      = [A_bool, A_real]
    loss_list   = [HingeLoss, QuadraticLoss]

To improve generalization error, we choose to use quadratic regularization 
on both factors X and Y with weight 0.1. (For no regularization on X and Y, use
ZeroReg.)

    from glrm.reg import QuadraticReg
    regX, regY = QuadraticReg(0.1), QuadraticReg(0.1)

If any entries are corrupted or missing, we stored indices of the missing
entries *for each submatrix* in the list format shown above. 
For example, if a 4x4 block of data is missing from the center of our example
above, this corresponds to rows 24-27 and columns 49-50 for submatrix 1,
and rows 24-27 and columns 1-2 for submatrix 2. (Python is 0-indexed.)

    missing1     = [(23, 48), (23, 49), (24, 48), (24, 49), \
                    (25, 48), (25, 49), (26, 48), (26, 49)]
    missing2     = [(23, 0), (23, 1), (24, 0), (24, 1), \
                    (25, 0), (25, 1), (26, 0), (26, 1)]
    missing_list = [missing1, missing2]

[Optional] To specify the tolerance and maximum number of iterations 
of the alternating minimization algorithm, create a Convergence object to pass
to the model. The default parameter values are shown below.

    from glrm.util import Convergence
    c = Convergence(TOL = 1e-3, max_iters = 1000)

All that remains is to initialize the GLRM model and call fit().

    model = GLRM(A_list, loss_list, regX, regY, k, converge = c)
    moel.fit()

To extract the factors X, Y and impute missing values,

    X, Y = model.factors()
    A_hat = glrm_mix.predict() # a horizontally concatenated matrix, not a list

To compare our prediction error,
    
    norm(A_hat - hstack(A_list)) # by hand

To view convergence history,

    ch = model.convergence() # grab convergence history of alt min problem
    ch.plot() # view convergence of objective


## Supported loss functions and regularizers

 - QuadraticLoss
 - HuberLoss
 - HingeLoss
 - OrdinalLoss


 - ZeroReg
 - LinearReg
 - QuadraticReg

## Developing loss functions and regularizers (not guaranteed to work yet)

 - FractionalLoss
 - NonnegativeReg

To use NonnegativeReg on either X or Y, you must specify to use proximal
gradient descent on the corresponding subproblem.

    # given A_list, loss_list, k from above
    from glrm.reg import NonnegativeReg
    from glrm.algs import ProxGD

    regX, regY = NonnegativeReg(1.0), NonnegativeReg(1.0)
    model = GLRM(A_list, loss_list, regX, regY, k, algX = ProxGD, algY = ProxGD)
    model.fit()


## Questions/concerns/feedback
Please send messages to Corinne Horn (cehorn at stanford.edu).
