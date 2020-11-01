import numpy as np
# import matplotlib.pyplot as plt
from cpp2g import CPP2
from scipy.io import loadmat
import argparse
from scipy.spatial.distance import pdist, squareform, cdist



parser = argparse.ArgumentParser(description = 'This scripts fits the CPP model to data from the Goglia 2019 drug screen data')


parser.add_argument('--min', dest = 'min', type = int, default = 0, help = 'min well index')
parser.add_argument('--max', dest = 'max', type = int, default = 50, help = 'max arg index')

args = parser.parse_args()

dat = loadmat('/tigress/architv/drug-lfm/FINAL_EQ.mat')
params = np.zeros((450,7,3))
likelihoods = np.zeros((450,3))

sharpness = 1.
note = 'cpp 2g - kernel is fixed, epsilon adapts to neighbors is ~5, control'
kernel = 'fixed'


def set_epsilon(well, thresh = np.arange(10,110,1)):
    cxs = np.zeros(len(thresh))
    x = dat['well'][0][well]['x']
    y = dat['well'][0][well]['y']
    coords = np.vstack((x[0],y[0])).T
    d = squareform(pdist(coords))
    for j in range(len(thresh)):
        cxs[j] = ((np.sum(d < thresh[j]) - x.shape[1])/x.shape[1])
    return thresh[np.argmin(np.square(cxs - 5))], cxs[np.argmin(np.square(cxs - 5))]


for i in range(args.min,args.max):
    print(i)
    well = i
    t = dat['well'][0][well]['t']
    traces = dat['well'][0][well]['PeakTimes']
    events = []
    for j in range(len(traces[0])):
        events.append(np.reshape(traces[0][j],-1).astype(np.float64))
    x = dat['well'][0][well]['x']
    y = dat['well'][0][well]['y']
    if i == 201:
        x = x[:,:146]
        y = y[:,:146]
    coords = np.vstack((x[0],y[0])).T
    
    lpp = CPP2()
    
    ep, n = set_epsilon(well)
    #ep = 60.
    #n = 0
    
    lpp.input_data(events, coords, np.zeros(coords.shape[0], dtype = np.int))
    lpp.set_init(nclusters = 1, mu_init = np.ones((1,)), a_init = np.ones((1,1)), 
                 b_init = np.ones((1,1)),
                 epsilon_init = ep, tmax = 120)
    lpp.fit_with_self(10000, sharpness = sharpness, lr = 1e-4, kernel = kernel) #, control = True)
    
    params[i,0,0] = lpp.get_mu_learned()
    params[i,1,0] = lpp.get_a_learned()
    params[i,2,0] = lpp.get_b_learned()
    params[i,3,0] = lpp.get_ajj_learned()
    params[i,4,0] = lpp.get_bjj_learned()
    
    params[i,5,0] = ep
    params[i,6,0] = n # average # neighbors
    likelihoods[i,0] = lpp.loss
    print(' ')
    
#     lpp.input_data(events, coords, np.zeros(coords.shape[0], dtype = np.int))
#     lpp.set_init(nclusters = 1, tmax = 120)#mu_init = mu_init, 
                 #a_init = a_init, 
                 #ajj_init = ajj_init, 
                 #b_init = b_init, 
                 #epsilon_init = epsilon_init)
    
#     lpp.fit_control(10000, sharpness = sharpness, lr = 1e-4)
    
#     params[i,0,1] = lpp.get_mu_learned()
#     params[i,1,1] = lpp.get_a_learned()
#     params[i,2,1] = lpp.get_b_learned()
#     params[i,3,1] = lpp.get_epsilon_learned()
#     #params[i,4,1] = lpp.get_ajj_learned()
#     likelihoods[i,1] = lpp.loss_control
#     print(' ')
    
    
#     lpp.set_init(nclusters = 1, epsilon_init = 15)#mu_init = mu_init, 
#                  #a_init = a_init, 
#                  #ajj_init = ajj_init, 
#                  #b_init = b_init, 
#                  #epsilon_init = epsilon_init)
#     lpp.input_data(events, coords, np.zeros(coords.shape[0], dtype = np.int))
#     lpp.fit_spatial(10000, sharpness = sharpness, lr = 1e-4)
    
#     params[i,0,2] = lpp.get_mu_learned()
#     params[i,1,2] = lpp.get_a_learned()
#     params[i,2,2] = lpp.get_b_learned()
#     params[i,3,2] = lpp.get_epsilon_learned()
#     params[i,4,2] = lpp.get_ajj_learned()
#     likelihoods[i,2] = lpp.loss_spatial
#     print(' ')
    
fname = './v6-cpp2g/with_self/drug/params-control-' + kernel + '-' + str(args.min) + '-' + str(args.max)
np.savez(fname, params = params, likelihoods = likelihoods, note = note)
