An overview of each of the simulations in this folder.

pca_nucnorm.py
 - numerical data
 - QuadraticLoss, QuadraticReg
 - second example labels center as missing

pca_nonneg.py (still buggy with prox gradient descent tuning parameters)
 - numerical data
 - QuadraticLoss, NonnegativeReg

huber_pca.py
 - approximately low rank numerical data with sparse, high-magnitude noise
 - HuberLoss, QuadraticReg
 - second example labels center as missing

likert.py
 - data table with ordinal data (1-7 integer data) 
 - OrdinalLoss, QuadraticReg

mixed.py
 - data table containing numerical, ordinal (1-7 integer), and Boolean data
 - (QuadraticLoss, OrdinalLoss, HingeLoss), QuadraticReg
 - second example labels lower right block as missing

smiley.py (somewhat slow)
 - Boolean data (in the shape of a smiley face)
 - HingeLoss, QuadraticReg 

fractional_pca.py (does not work)
 - for testing the development of FractionalLoss
