import numpy as np
from cpp2g import CPP2
#from matplotlib import animation, rc
#import matplotlib.pyplot as plt
#from IPython.display import HTML
import time
from scipy.stats import pearsonr, spearmanr
import argparse


parser = argparse.ArgumentParser(description = 'This scripts fits the CPP model to simulated data across cell/peak pairs')
parser.add_argument('--seed', dest = 'seed', type = int, default = 0, help = 'random seed + index of run')
args = parser.parse_args()


out = './v6-cpp2g/with_self/frontier4/'
run = str(args.seed)

np.random.seed(args.seed)

cells = [50, 75, 100, 125, 150, 175, 200]
ncells = len(cells)

peaks = [100, 250, 500, 1000, 2500]
npeaks = len(peaks)

init_test = 1

nsims_per_condition = 1

a = np.random.uniform(0.01, 1.5, size = (1,1)) #1. * np.ones((1,1))
ajj = np.random.uniform(0.01, 1.5, size = (1,))
b = np.random.uniform(0.01, 5, size = (1,1)) #1. * np.ones((1,1))
bjj = np.random.uniform(0.01, 1.5, size = (1,))
mu = np.random.uniform(0.01, 3, size = (1,)) #1. * np.ones((1,))

print(a)
print(ajj)
print(b)
print(bjj)
print(mu)

epsilon = 60.

learned_a = np.zeros((ncells, npeaks, nsims_per_condition, init_test, 1, 1))
learned_b = np.zeros((ncells, npeaks, nsims_per_condition, init_test, 1, 1))
learned_mu = np.zeros((ncells, npeaks, nsims_per_condition, init_test, 1))
learned_ajj = np.zeros((ncells, npeaks, nsims_per_condition, init_test, 1))
learned_bjj = np.zeros((ncells, npeaks, nsims_per_condition, init_test, 1))
learned_t = np.zeros((ncells, npeaks, nsims_per_condition))
time_to_fit = np.zeros((ncells, npeaks, nsims_per_condition,init_test))

peaks_sim = np.zeros((ncells, npeaks), dtype = object)
coords = np.zeros((ncells, npeaks), dtype = object)

for i in range(ncells):
    for j in range(npeaks):
        for k in range(nsims_per_condition):
            cpp = CPP2()
            width = cells[i] * np.pi * 600.
            cpp.simulate_with_self(N = cells[i], 
                                   tmax = 100,
                                   peakmax = peaks[j],
                                   nclusters = 1, 
                                   p = [1.],
                                   xmax = width, 
                                   mu = mu,
                                   a = a, ajj = ajj,
                                   b = b, bjj = bjj,
                                   epsilon =  60)
            
            peaks_sim[i,j] = cpp.peaks
            coords[i,j] = cpp.coords
            
            learned_t[i,j,k] = cpp.tmax
            for l in range(init_test):
                #if l == 0:
                #    cpp.set_init(mu_init = None, a_init = None, 
                #                 b_init = None, epsilon_init = 60.)

                #else:
                cpp.set_init(mu_init = np.ones((1,)), a_init = np.ones((1,1)), 
                             b_init = np.ones((1,1)), ajj_init = np.ones((1,)),
                             bjj_init = np.ones((1,)), epsilon_init = 60.)
                    
                t1 = time.perf_counter()
                cpp.fit_with_self(iters = 5000, kernel = 'fixed', lr = 1e-4)
                t2 = time.perf_counter()
                
                learned_a[i,j,k,l] = cpp.get_a_learned()
                learned_b[i,j,k,l] = cpp.get_b_learned()
                #learned_b_mean[i,j,k,l] = cpp.get_b_mean_learned()
                learned_mu[i,j,k,l] = cpp.get_mu_learned()
                learned_ajj[i,j,k,l] = cpp.get_ajj_learned()
                learned_bjj[i,j,k,l] = cpp.get_bjj_learned()
                time_to_fit[i,j,k,l] = t2 - t1

np.savez(out + '/learned-' + run,
         t = learned_t,
         mu = learned_mu, 
         a = learned_a, 
         b = learned_b, 
         ajj = learned_ajj,
         bjj = learned_bjj,
         true_mu = mu,
         true_a = a, true_ajj = ajj,
         true_b = b, true_bjj = bjj,
         fit_time = time_to_fit,
         peaks = peaks_sim, coords = coords)