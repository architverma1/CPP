from cpp2g import CPP2
from scipy.io import loadmat
import numpy as np
from scipy.stats import pearsonr, spearmanr

params = np.zeros((24,5,2))
likelihoods = np.zeros((24,2))

date = '102920'
dat = loadmat('/tigress/architv/drug-lfm/sidu-frontier/dataforcpp.mat')

note = 'cpp2g - epsilon = 60'

nsims = 24
nclusters = 2

iters = 1000
kernels = ['fixed']

mu_learned = np.zeros((1,nsims,nclusters))
a_learned = np.zeros((1,nsims,nclusters,nclusters))
ajj_learned = np.zeros((1,nsims,nclusters))
#b_mean_learned = np.zeros((1,nsims,nclusters))
b_learned = np.zeros((1,nsims,nclusters,nclusters))
bjj_learned = np.zeros((1,nsims,nclusters))
epsilon_learned = np.zeros((1,nsims))
likelihood_learned = np.zeros((1,nsims))

mu_learned_prior = np.zeros((nsims,nclusters))
a_learned_prior = np.zeros((nsims,nclusters,nclusters))
#b_mean_learned_prior = np.zeros((nsims,nclusters))
b_learned_prior = np.zeros((nsims,nclusters,nclusters))
bjj_learned_prior = np.zeros((nsims,nclusters))
epsilon_learned_prior = np.zeros(nsims)
likelihood_learned_prior = np.zeros(nsims)

for i in range(24):
    print(i)
    well = i
    traces_ktr = dat['wells_final'][0][well]['PeakTimesKTR']
    traces_gfp = dat['wells_final'][0][well]['PeakTimesGFP']
    traces = np.hstack((traces_ktr, traces_gfp))
    N = len(traces_ktr[0])
    
    x = np.hstack((dat['wells_final'][0][well]['x'],dat['wells_final'][0][well]['x']))
    y = np.hstack((dat['wells_final'][0][well]['y'],dat['wells_final'][0][well]['y']))
    
    events = []
    for j in range(len(traces[0])):
        events.append(np.reshape(traces[0][j],-1).astype(np.float64))
        
    coords = np.vstack((x[0],y[0])).T
    cpp = CPP2()
    
    
    clusters = np.hstack((np.zeros(N),np.ones(N))).astype(np.int)
    cpp.input_data(events, coords, clusters)
    
    for j in range(len(kernels)):
        cpp.set_init(epsilon_init = 60, tmax = 480)
        cpp.fit_with_self(iters = iters, sharpness = 1., kernel = kernels[j])
        mu_learned[j,i] = cpp.get_mu_learned()
        a_learned[j,i] = cpp.get_a_learned()
        ajj_learned[j,i] = cpp.get_ajj_learned()
        b_learned[j,i] = cpp.get_b_learned()
        bjj_learned[j,i] = cpp.get_bjj_learned()
        epsilon_learned[j,i] = cpp.get_epsilon_learned()
        likelihood_learned[j,i] = cpp.loss
        print('')
        
    
np.savez('./v6-cpp2g/with_self/gfp/learned-' + date, 
         mu = mu_learned, a = a_learned, 
         ajj = ajj_learned,
         bjj = bjj_learned, b = b_learned,
         epsilon = epsilon_learned, likelihood = likelihood_learned, note = note)
        
