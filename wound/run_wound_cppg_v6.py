from scipy.io import loadmat
import numpy as np
from cpp2g import CPP2
#CHANGES MADE: in v3 the epsilon was changed from 60 to 80 to get more associations. also, a speed measurement was added.
x = loadmat('datadrugsnorm.mat')
boxes_x = 10
boxes_y = 5
#first dmso:
xx = x['wellnormdmso'][0]['x'][0]
#xx = np.divide(xx, np.max(xx))
#xx.shape
yy = x['wellnormdmso'][0]['y'][0]
#yy = np.divide(yy, np.max(yy))
#yy.shape
coords = np.stack((xx[0],yy[0])).T


incrementx = np.max(xx)/boxes_x
incrementy = np.max(yy)/boxes_y


params_x = np.zeros((boxes_x,6,2))
params_y = np.zeros((boxes_y,6,2))

likelihoods_x = np.zeros((boxes_x,2))
likelihoods_y = np.zeros((boxes_y,2))

speeds_x = np.zeros((boxes_x, 1))

sharpness = 1.

for i in range(boxes_x):
    low = i * incrementx
    high = (i + 1) * incrementx
    mask = (coords[:,0] > low) & (coords[:,0] < high)
    if np.sum(mask) == 0:
	    continue
    traces = x['wellnormdmso'][0]['PeakTimes'][0][:,mask]
    peaks = []
    for j in range(len(traces[0])):
        peaks.append(np.reshape(traces[0][j],-1).astype(np.float64))
        
    lpp = CPP2()
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(epsilon_init = 80, tmax = 360, b_init = 10*np.ones((1,1)))
    



    lpp.fit_with_self(10000, sharpness = sharpness, lr = 1e-4, kernel = 'fixed')


    params_x[i,0,0] = lpp.get_mu_learned()
    params_x[i,1,0] = lpp.get_a_learned()
    params_x[i,2,0] = lpp.get_b_learned()
    params_x[i,3,0] = lpp.get_epsilon_learned()
    params_x[i,4,0] = lpp.get_ajj_learned()
    params_x[i,5,0] = lpp.get_bjj_learned ()
    likelihoods_x[i,0] = lpp.loss


    
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(nclusters = 1)#mu_init = mu_init, 
                 #a_init = a_init, 
                 #ajj_init = ajj_init, 
                 #b_init = b_init, 
                 #epsilon_init = epsilon_init)
    lpp.fit_with_self(1000, sharpness = sharpness, lr = 1e-4, control = True)
    
    params_x[i,0,1] = lpp.get_mu_learned()
    params_x[i,1,1] = lpp.get_a_learned()
    params_x[i,2,1] = lpp.get_b_learned()
    params_x[i,3,1] = lpp.get_epsilon_learned()
    params_x[i,4,1] = lpp.get_ajj_learned ()
    params_x[i,5,1] = lpp.get_bjj_learned ()
    likelihoods_x[i,1] = lpp.loss

    xdif = xx[0 ,mask]-xx[360, mask]
    speeds_x[i, 0] = np.mean(xdif)

    print(' ')
    
fname = 'params-dmso-exp-g-x' + str(boxes_x)
np.savez(fname, params = params_x, likelihoods = likelihoods_x, speeds = speeds_x)


for i in range(boxes_y):
    low = i * incrementy
    high = (i + 1) * incrementy
    mask = (coords[:,1] > low) & (coords[:,1] < high)
    if np.sum(mask) == 0:
	    continue
    traces = x['wellnormdmso'][0]['PeakTimes'][0][:,mask]
    peaks = []
    for j in range(len(traces[0])):
        peaks.append(np.reshape(traces[0][j],-1).astype(np.float64))
        
    lpp = CPP2()
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(epsilon_init = 80, tmax = 480)
    



    lpp.fit_with_self(10000, sharpness = sharpness, lr = 1e-4, kernel = 'fixed')


    params_y[i,0,0] = lpp.get_mu_learned()
    params_y[i,1,0] = lpp.get_a_learned()
    params_y[i,2,0] = lpp.get_b_learned()
    params_y[i,3,0] = lpp.get_epsilon_learned()
    params_y[i,4,0] = lpp.get_ajj_learned()
    params_y[i,5,0] = lpp.get_bjj_learned ()
    likelihoods_y[i,0] = lpp.loss
    
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(nclusters = 1)#mu_init = mu_init, 
                 #a_init = a_init, 
                 #ajj_init = ajj_init, 
                 #b_init = b_init, 
                 #epsilon_init = epsilon_init)
    lpp.fit_with_self(1000, sharpness = sharpness, lr = 1e-4, control = True)
    
    params_y[i,0,1] = lpp.get_mu_learned()
    params_y[i,1,1] = lpp.get_a_learned()
    params_y[i,2,1] = lpp.get_b_learned()
    params_y[i,3,1] = lpp.get_epsilon_learned()
    params_y[i,4,1] = lpp.get_ajj_learned ()
    params_y[i,5,1] = lpp.get_bjj_learned ()
    likelihoods_y[i,1] = lpp.loss
    print(' ')
    
fname = 'params-dmso-exp-g-y' + str(boxes_y)
np.savez(fname, params = params_y, likelihoods = likelihoods_y)



print("DMSO finished!")



#next tapi:
xx = x['wellnormtapi'][0]['x'][0]
#xx = np.divide(xx, np.max(xx))
#xx.shape
yy = x['wellnormtapi'][0]['y'][0]
#yy = np.divide(yy, np.max(yy))
#yy.shape
coords = np.stack((xx[0],yy[0])).T


incrementx = np.max(xx)/boxes_x
incrementy = np.max(yy)/boxes_y


params_x = np.zeros((boxes_x,6,2))
params_y = np.zeros((boxes_y,6,2))

likelihoods_x = np.zeros((boxes_x,2))
likelihoods_y = np.zeros((boxes_y,2))

speeds_x = np.zeros((boxes_x, 1))

sharpness = 1.

for i in range(boxes_x):
    low = i * incrementx
    high = (i + 1) * incrementx
    mask = (coords[:,0] > low) & (coords[:,0] < high)
    if np.sum(mask) == 0:
	    continue
    traces = x['wellnormtapi'][0]['PeakTimes'][0][:,mask]
    peaks = []
    for j in range(len(traces[0])):
        peaks.append(np.reshape(traces[0][j],-1).astype(np.float64))
        
    lpp = CPP2()
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(epsilon_init = 80, tmax = 480, b_init = 10*np.ones((1,1)))
    



    lpp.fit_with_self(10000, sharpness = sharpness, lr = 1e-4, kernel = 'fixed')


    params_x[i,0,0] = lpp.get_mu_learned()
    params_x[i,1,0] = lpp.get_a_learned()
    params_x[i,2,0] = lpp.get_b_learned()
    params_x[i,3,0] = lpp.get_epsilon_learned()
    params_x[i,4,0] = lpp.get_ajj_learned()
    params_x[i,5,0] = lpp.get_bjj_learned()
    likelihoods_x[i,0] = lpp.loss
    
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(nclusters = 1)#mu_init = mu_init, 
                 #a_init = a_init, 
                 #ajj_init = ajj_init, 
                 #b_init = b_init, 
                 #epsilon_init = epsilon_init)
    lpp.fit_with_self(1000, sharpness = sharpness, lr = 1e-4, control = True)
    
    params_x[i,0,1] = lpp.get_mu_learned()
    params_x[i,1,1] = lpp.get_a_learned()
    params_x[i,2,1] = lpp.get_b_learned()
    params_x[i,3,1] = lpp.get_epsilon_learned()
    params_x[i,4,1] = lpp.get_ajj_learned ()
    params_x[i,5,1] = lpp.get_bjj_learned()
    likelihoods_x[i,1] = lpp.loss

    xdif = xx[0 ,mask]-xx[360, mask]
    speeds_x[i, 0] = np.mean(xdif)

    print(' ')
    
fname = 'params-tapi-exp-g-x' + str(boxes_x)
np.savez(fname, params = params_x, likelihoods = likelihoods_x, speeds = speeds_x)


for i in range(boxes_y):
    low = i * incrementy
    high = (i + 1) * incrementy
    mask = (coords[:,1] > low) & (coords[:,1] < high)
    if np.sum(mask) == 0:
	    continue
    traces = x['wellnormtapi'][0]['PeakTimes'][0][:,mask]
    peaks = []
    for j in range(len(traces[0])):
        peaks.append(np.reshape(traces[0][j],-1).astype(np.float64))
        
    lpp = CPP2()
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(epsilon_init = 80, tmax = 480)
    



    lpp.fit_with_self(10000, sharpness = sharpness, lr = 1e-4, kernel = 'fixed')


    params_y[i,0,0] = lpp.get_mu_learned()
    params_y[i,1,0] = lpp.get_a_learned()
    params_y[i,2,0] = lpp.get_b_learned()
    params_y[i,3,0] = lpp.get_epsilon_learned()
    params_y[i,4,0] = lpp.get_ajj_learned()
    params_y[i,5,0] = lpp.get_bjj_learned()
    likelihoods_y[i,0] = lpp.loss
    
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(nclusters = 1)#mu_init = mu_init, 
                 #a_init = a_init, 
                 #ajj_init = ajj_init, 
                 #b_init = b_init, 
                 #epsilon_init = epsilon_init)
    lpp.fit_with_self(1000, sharpness = sharpness, lr = 1e-4, control = True)
    
    params_y[i,0,1] = lpp.get_mu_learned()
    params_y[i,1,1] = lpp.get_a_learned()
    params_y[i,2,1] = lpp.get_b_learned()
    params_y[i,3,1] = lpp.get_epsilon_learned()
    params_y[i,4,1] = lpp.get_ajj_learned ()
    params_y[i,5,1] = lpp.get_bjj_learned()
    likelihoods_y[i,1] = lpp.loss
    print(' ')
    
fname = 'params-tapi-exp-g-y' + str(boxes_y)
np.savez(fname, params = params_y, likelihoods = likelihoods_y)



print("TAPI finished!")



xx = x['wellnormtram'][0]['x'][0]
#xx = np.divide(xx, np.max(xx))
#xx.shape
yy = x['wellnormtram'][0]['y'][0]
#yy = np.divide(yy, np.max(yy))
#yy.shape
coords = np.stack((xx[0],yy[0])).T


incrementx = np.max(xx)/boxes_x
incrementy = np.max(yy)/boxes_y


params_x = np.zeros((boxes_x,6,2))
params_y = np.zeros((boxes_y,6,2))

likelihoods_x = np.zeros((boxes_x,2))
likelihoods_y = np.zeros((boxes_y,2))

speeds_x = np.zeros((boxes_x, 1))

sharpness = 1.

for i in range(boxes_x):
    low = i * incrementx
    high = (i + 1) * incrementx
    mask = (coords[:,0] > low) & (coords[:,0] < high)
    if np.sum(mask) == 0:
	    continue
    traces = x['wellnormtram'][0]['PeakTimes'][0][:,mask]
    peaks = []
    for j in range(len(traces[0])):
        peaks.append(np.reshape(traces[0][j],-1).astype(np.float64))
        
    lpp = CPP2()
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(epsilon_init = 80, tmax = 480, b_init = 10*np.ones((1,1)))
    



    lpp.fit_with_self(10000, sharpness = sharpness, lr = 1e-4, kernel = 'fixed')


    params_x[i,0,0] = lpp.get_mu_learned()
    params_x[i,1,0] = lpp.get_a_learned()
    params_x[i,2,0] = lpp.get_b_learned()
    params_x[i,3,0] = lpp.get_epsilon_learned()
    params_x[i,4,0] = lpp.get_ajj_learned()
    params_x[i,5,0] = lpp.get_bjj_learned()
    likelihoods_x[i,0] = lpp.loss
    
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(nclusters = 1)#mu_init = mu_init, 
                 #a_init = a_init, 
                 #ajj_init = ajj_init, 
                 #b_init = b_init, 
                 #epsilon_init = epsilon_init)
    lpp.fit_with_self(1000, sharpness = sharpness, lr = 1e-4, control = True)
    
    params_x[i,0,1] = lpp.get_mu_learned()
    params_x[i,1,1] = lpp.get_a_learned()
    params_x[i,2,1] = lpp.get_b_learned()
    params_x[i,3,1] = lpp.get_epsilon_learned()
    params_x[i,4,1] = lpp.get_ajj_learned ()
    params_x[i,5,1] = lpp.get_bjj_learned ()
    likelihoods_x[i,1] = lpp.loss

    xdif = xx[0 ,mask]-xx[360, mask]
    speeds_x[i, 0] = np.mean(xdif)
    print(' ')
    
fname = 'params-tram-exp-g-x' + str(boxes_x)
np.savez(fname, params = params_x, likelihoods = likelihoods_x, speeds = speeds_x)


for i in range(boxes_y):
    low = i * incrementy
    high = (i + 1) * incrementy
    mask = (coords[:,1] > low) & (coords[:,1] < high)
    if np.sum(mask) == 0:
	    continue
    traces = x['wellnormtram'][0]['PeakTimes'][0][:,mask]
    peaks = []
    for j in range(len(traces[0])):
        peaks.append(np.reshape(traces[0][j],-1).astype(np.float64))
        
    lpp = CPP2()
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(epsilon_init = 80, tmax = 480)
    



    lpp.fit_with_self(10000, sharpness = sharpness, lr = 1e-4, kernel = 'fixed')


    params_y[i,0,0] = lpp.get_mu_learned()
    params_y[i,1,0] = lpp.get_a_learned()
    params_y[i,2,0] = lpp.get_b_learned()
    params_y[i,3,0] = lpp.get_epsilon_learned()
    params_y[i,4,0] = lpp.get_ajj_learned()
    params_y[i,5,0] = lpp.get_bjj_learned()
    likelihoods_y[i,0] = lpp.loss
    
    lpp.input_data(peaks, coords[mask], np.zeros(coords[mask].shape[0], dtype = np.int))
    lpp.set_init(nclusters = 1)#mu_init = mu_init, 
                 #a_init = a_init, 
                 #ajj_init = ajj_init, 
                 #b_init = b_init, 
                 #epsilon_init = epsilon_init)
    lpp.fit_with_self(1000, sharpness = sharpness, lr = 1e-4, control = True)
    
    params_y[i,0,1] = lpp.get_mu_learned()
    params_y[i,1,1] = lpp.get_a_learned()
    params_y[i,2,1] = lpp.get_b_learned()
    params_y[i,3,1] = lpp.get_epsilon_learned()
    params_y[i,4,1] = lpp.get_ajj_learned ()
    params_y[i,5,1] = lpp.get_bjj_learned ()
    likelihoods_y[i,1] = lpp.loss
    print(' ')
    
fname = 'params-tram-exp-g-y' + str(boxes_y)
np.savez(fname, params = params_y, likelihoods = likelihoods_y)



print("Trametinib finished!")