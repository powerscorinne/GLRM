from paths import *
from glrm import GLRM, squared_loss, hinge_loss, ordinal_loss, zero_reg, \
        norm1_reg, norm2sq_reg
from numpy import hstack, loadtxt, mean
from numpy.linalg import norm
from numpy.random import seed
from matplotlib import pyplot as plt
from glrm.utils.pretty_plot import visualize_recovery, draw_latent_space
from random import sample
from random import seed as seedr
seed(44)
seedr(17)

## ================= Problem data ==========================
max_rows = 5914 # truncate number of questionnaires we load #5914
k = 3
responses_file = data_dir + "/anes/version4/processed_anes_responses_4.txt"
questions_file = data_dir + "/anes/version4/processed_anes_questions_4.txt" 

data = loadtxt(responses_file)
qid, prompt = [], [] # question id, question prompt

num_indx = [] # indices of numerical data
resp_indx = {} # indices of categorical (likert) data
for i in range(2,10): resp_indx[i] = [] # keep track of number of possible responses

n = 0 # question number
with open(questions_file) as f: # load data
    for line in f:
        (q, p, r) = line.split(" | ")
        qid.append(q)
        prompt.append(p)
        if max(eval(r)) >= 10: num_indx.append(n)
        else: resp_indx[max(eval(r))].append(n)
        n += 1

A1 = hstack([data[:, indx:indx+1] for indx in num_indx]) # numerical data
qid_indx = num_indx
As = [A1]
for v in resp_indx.values():
    As.append(hstack([data[:, indx:indx+1] for indx in v])) 
    qid_indx = hstack((qid_indx, v))
qid = [qid[i] for i in qid_indx]

# manual for now (human-readable version of qid)
qid2 = ['age', 'income', 'health insured', 'has smartphone', 'will vote for (president)', 
        'voted for (president)', 'ethnicity', 'social class',
        'patriotic', 'flag appreciation', 'education', 'health', 
        'marital status', 'health care bill', 'women in workplace', 
        'aa (university)', 'aa (work)', 'employed', 'abortion rights']


# truncate rows for testing
As = [As[i][:max_rows,:] for i in range(len(As))]

## ================= Missing data ==================
missing = []
for a in range(len(As)):
    missingA = []
    for i in range(As[a].shape[0]):
        for j in range(As[a].shape[1]):
            if As[a][i,j] < 0:
                missingA.append((i,j))
    missing.append(missingA)
ages_removed = sample([i for i, a in enumerate(As[0][:,1]) if a >= 0],
        int(max_rows*.01))
for i in ages_removed: missing[0].append((i, 1))

## ============== Loss functions, regularizers ==================
losses = [squared_loss]
for i in range(len(As) - 1): losses.append(ordinal_loss)
regsY, regX = norm2sq_reg(0.1), norm2sq_reg(.1)

## ============== Model =================================
model = GLRM(As, losses, regsY, regX, k, missing)
At = model.alt_min(outer_RELTOL = 1e-1, inner_RELTOL = 1e-4, outer_max_iters =
        10, inner_max_iters = 100, quiet = False)

## ================= Results ===========================
Am = hstack((A for A in As)) # matrix version of As
Atm = hstack((A for A in At)) # matrix version of At
X = model.X
Y = hstack([ym for ym in model.Y])

# visualizations
color_markers = data[:max_rows, -2] # vote for 2012 president (only when both = True)
plotting_filter = [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
Y = Y[:,plotting_filter]
qid2 = [qid2[i] for i in plotting_filter]
draw_latent_space(X, Y, qid2, 1, color_markers, both = False) 
