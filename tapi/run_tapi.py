from cpp2g import CPP2
from scipy.io import loadmat
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

params = np.zeros((20,5,2))
likelihoods = np.zeros((20,2))

date = '103120'

dat = loadmat('/tigress/architv/ORGANIZED_CODE_FOR_UPLOAD/tapi_dose.mat')
note = 'cpp2g - epsilon = 60, init with mu = half control mu, a = half control mu, b = .01*ones, bjj = 1, model with self'


nsims = 20
nclusters = 1

mu_learned = np.zeros((3,nsims,nclusters))
a_learned = np.zeros((3,nsims,nclusters,nclusters))
ajj_learned = np.zeros((3,nsims,nclusters))
b_learned = np.zeros((3,nsims,nclusters,nclusters))
bjj_learned = np.zeros((3,nsims,nclusters))
epsilon_learned = np.zeros((3,nsims))
likelihood_learned = np.zeros((3,nsims))

mu_learned_prior = np.zeros((nsims,nclusters))
a_learned_prior = np.zeros((nsims,nclusters,nclusters))
ajj_learned_prior = np.zeros((nsims,nclusters))
b_learned_prior = np.zeros((nsims,nclusters,nclusters))
bjj_learned_prior = np.zeros((nsims,nclusters))
epsilon_learned_prior = np.zeros(nsims)
likelihood_learned_prior = np.zeros(nsims)

mu_learned_spatial = np.zeros((3,nsims,nclusters))
a_learned_spatial = np.zeros((3,nsims,nclusters,nclusters))
ajj_learned_spatial = np.zeros((3,nsims,nclusters))
b_learned_spatial = np.zeros((3,nsims,nclusters,nclusters))
bjj_learned_spatial = np.zeros((3,nsims,nclusters))
epsilon_learned_spatial = np.zeros((3,nsims))
likelihood_learned_spatial = np.zeros((3,nsims))

iters = 2000
kernels = ['fixed']


for i in range(1,20):
    print(i)
    well = i
    t = dat['well'][0][well]['t']
    traces = dat['well'][0][well]['PeakTimes']
    events = []
    for j in range(len(traces[0])):
        events.append(np.reshape(traces[0][j],-1).astype(np.float64))
    x = dat['well'][0][well]['x']
    y = dat['well'][0][well]['y']
    coords = np.vstack((x[0],y[0])).T
    cpp = CPP2()
    cpp.input_data(events, coords, np.zeros((coords.shape[0],), dtype = np.int))
    

    cpp.set_init(ajj_init = np.ones((1,)),
                 b_init = .01 * np.ones((1,1)), 
                 bjj_init = np.ones((1,)),
                 epsilon_init = 60, tmax = 480)
    cpp.fit_with_self(iters = iters, kernel = 'fixed')
    mu_learned[0,i] = cpp.get_mu_learned()
    a_learned[0,i] = cpp.get_a_learned()
    ajj_learned[0,i] = cpp.get_ajj_learned()
    b_learned[0,i] = cpp.get_b_learned()
    bjj_learned[0,i] = cpp.get_bjj_learned()
    epsilon_learned[0,i] = cpp.get_epsilon_learned()
    likelihood_learned[0,i] = cpp.loss
        
    
    
np.savez('./v6-cpp2g/with_self/tapi/learned-' + date, 
         mu = mu_learned, a = a_learned, 
         ajj = ajj_learned, b = b_learned, bjj = bjj_learned,
         epsilon = epsilon_learned, likelihood = likelihood_learned, note = note)
