import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import lognorm

import torch
from torch.distributions.log_normal import LogNormal
from torch.distributions.gamma import Gamma
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.cauchy import Cauchy
from torch.distributions.normal import Normal

import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation, rc
plt.style.use('seaborn-poster')

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_inv(x):
    return np.log(np.exp(x) - 1)

def logit(x):
    return np.log(x/(1.-x))

def log_normal_pdf(y,b):
    y[y == 0] = 1e-10
    yb = torch.div(torch.log(y),b)
    expyb = torch.exp(-0.5 * torch.pow(yb,2))
    frac = torch.div(1, torch.mul(y,b) * np.sqrt(2*np.pi))
    return torch.mul(frac, expyb)

def log_normal_cdf(y,b):
    y[y == 0] = 1e-10
    yb = torch.div(torch.log(y), np.sqrt(2) * b)
    #yb = y
    #print(yb)
    fx = torch.erf(yb) #erf_approximation(yb)
    #fx = yb
    #print(fx.grad)
    #print(fx)
    return 0.5 + 0.5*fx

def erf_approximation(x):
    p = 0.47047
    a1 = 0.3480242
    a2 = -0.0958798
    a3 = 0.7478556
    
    t = torch.div(1, 1 + p*x)
    
    return 1 - torch.mul(torch.exp(-torch.pow(x,2)), a1 * t + a2 * torch.pow(x,2) + a3 * torch.pow(x,3))

class CPP2:
    
    def __init__(self):
        return
    
    def simulate(self, N = 100, 
                 tmax = 16,
                 peakmax = 5000,
                 nclusters = 3, 
                 p = None,
                 xmax = 700, 
                 mu = None,
                 a = None, 
                 b = None,
                 epsilon =  None):
        
        if mu is None:
            #mu = 10.** np.random.uniform(-2,0.3, size = nclusters)
            mu = np.random.uniform(0, 2, size = nclusters)
        
        if a is None:
            a = np.random.uniform(0, 2, size = (nclusters,nclusters))
            
            
        if b is None:
            b = np.random.uniform(0,4, size = (nclusters,nclusters))
            #np.multiply(10.**np.random.uniform(0,1,size = (nclusters, nclusters)), mu) # was -0.5 to 0.5 in test8
        
        if epsilon is None:
            epsilon = np.random.uniform(25,100)
            
        if p is None:
            p = np.random.dirichlet(np.ones(nclusters))
            
            
        coords = np.random.uniform(0,xmax,size=(N,2))
        clusters = np.random.choice(nclusters, size = N, p = p)
        d = squareform(pdist(coords))
        
        lambda_0 = mu[clusters] # array of size n cells with mu for each cell
        tnow = 0
        tlast = np.zeros(N) # intialize the last peak for each cell
        ti_all = np.random.exponential(1./lambda_0) #drawn the next peak for each cell
        
        peaks = [[]] * N # initialize array of array listing peaks
        adj = (d < epsilon) # define the cell-cell interactions
        
        tnow = np.min(ti_all) # move to next peak
        ts = np.expand_dims(np.array(tnow), axis = 0) # array formatting
        
        which_cell = np.expand_dims(np.array([np.argmin(ti_all)]), axis = 1) # move to cell
        
        ## calculate conditional intensity for each cell
        a_add = a[clusters[which_cell[-1]],clusters] # ajk for each cell, k is last cell
        #print(a_add.shape)
        b_add = b[clusters[which_cell[-1]],clusters] # b for each cell
        #print(b_add.shape)
        neg_bt = np.reshape(lognorm.pdf(tnow, b_add), -1)
        kernel = np.multiply(a_add, neg_bt) + lambda_0
        ti_all = np.random.exponential(1./kernel)
        
        while tnow < tmax:
            tnow = tnow + np.min(ti_all)
            #print('tnow')
            #print(tnow)
            
            cellnow = np.argmin(ti_all)
            #print('cellnow')
            #print(cellnow)
            ts = np.append(ts, tnow)
            ts = np.expand_dims(ts, axis = 0)
            which_cell = np.append(which_cell, cellnow)
            peaks[cellnow] = np.append(peaks[cellnow], tnow)

            ts_calc = np.repeat(ts, N, axis = 0)

            kernel = np.zeros(N)
            for i in range(len(which_cell)):
                a_add = np.multiply(a[clusters[which_cell[i]],clusters], adj[which_cell[i]])
                #print('a_add')
                #print(a_add)
                b_add = b[clusters[which_cell[i]],clusters]
                #print('b_add')
                #print(b_add)
                neg_bt = np.reshape(lognorm.pdf(tnow - ts[0,i], b_add), -1)
                #print('e^-bt')
                #print(neg_bt)
                kernel += np.multiply(a_add, neg_bt)
                #print('kernel')
                #print(kernel)
            #nu_sum = np.sum(np.multiply(ab, neg_bt), axis = 1)
            lambda_i = lambda_0 + kernel
            #print('lambda_i')
            #print(lambda_i)
            ti_all = np.random.exponential(1./lambda_i)
            #print('ti_all')
            #print(ti_all)
            if(len(ts[0])) > peakmax:
                print(len(ts[0])-1)
                break
        tnow = tnow + np.min(ti_all)  # one last draw to set tmax (length of time with no peaks observed)  
        self.peaks = peaks
        self.d = d
        self.coords = coords
        self.clusters = clusters
        self.a_true = a
        #self.ajj_true = ajj
        self.b_true = b

        self.mu_true = mu
        self.ep_true = epsilon
        self.nclusters = nclusters
        self.N = N
        self.tmax = tnow
        
        return
    
    def get_data(self):
        return self.peaks, self.coords, self.clusters
    
    def get_a_true(self):
        return self.a_true
    
    def get_ajj_true(self):
        return self.ajj_true 
    
    def get_b_true(self):
        return self.b_true
    
    def get_bjj_true(self):
        return self.bjj_true
    
    def get_mu_true(self):
        return self.mu_true
    
    def get_epsilon_true(self):
        return self.ep_true
    
    def get_N(self):
        return self.N
    
    def input_data(self, peaks, coords, clusters):
        self.peaks = peaks
        self.coords = coords
        self.N = len(peaks)
        self.clusters = clusters
        self.nclusters = len(np.unique(clusters))
        
        return
    
    def format_data(self):
        peaks = self.peaks
        y_flat = np.hstack(peaks)
        
        which_cell = np.zeros(len(y_flat))
        i0 = 0
        for i in range(len(peaks)):
            i1 = i0 + len(peaks[i])
            which_cell[i0:i1] = i
            i0 = i1
        
        self.y_sorted = np.sort(y_flat)
        self.cells_sorted = which_cell[np.argsort(y_flat)].astype(np.int)
        self.d = squareform(pdist(self.coords))
        
        
        return self.y_sorted, self.cells_sorted, self.d
    
    def negLogL(self, y_sorted, cells_sorted, d, clusters, 
                debug = False, a_prior = False, alpha = .001, beta = .001,
                kernel = None, control = False):
        
        sharpness = self.sharpness
        mu = torch.log(1 + torch.exp(self.mu_pre))
        a = torch.log(1 + torch.exp(self.a_pre))
        b = torch.log(1 + torch.exp(self.b_pre))
        epsilon = torch.log(1 + torch.exp(self.epsilon_pre))


        nc = len(y_sorted)
        N = len(self.peaks) #d.shape[0]
        D = torch.tensor(d)
        
        
        
        if kernel is None or kernel == 'logistic':
            adj = torch.div(torch.exp(-sharpness*(D - epsilon)), 1. + torch.exp(-sharpness*(D - epsilon)))
        elif kernel == 'relu':
            adj = (1./epsilon) * torch.nn.functional.relu(epsilon - D)
        
        elif kernel == 'exp':
            adj = torch.exp(-D/epsilon)
            
        elif kernel == 'fixed':
            adj = D < epsilon
            
        else:
            adj = torch.div(torch.exp(-sharpness*(D - epsilon)), 
                                                    1. + torch.exp(-sharpness*(D - epsilon)))
            
        if control:
            adj = torch.diag(torch.ones(N))
            
        
        #tmax = y_sorted[-1]

        logL = torch.tensor(0.)
        
        
        #logN = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        #logN = LogNormal(torch.tensor([0.0]), b)
        for i in range(1,nc):
            celli = cells_sorted[i].astype(np.int)
            #print(celli)
            whichi = cells_sorted[0:i].astype(np.int)
            #print(whichi)
            ai = a[clusters[celli],clusters[whichi]]
            bi = b[clusters[celli],clusters[whichi]]
            ti = y_sorted[0:i]
            yi = torch.tensor(y_sorted[i] - ti)
            #neg_bt = torch.exp(logN.log_prob(yi)) 
            neg_bt = log_normal_pdf(yi,bi)
            a_adj = torch.mul(ai, adj[celli, cells_sorted[0:i]])
            abg = torch.mul(a_adj, neg_bt)
            mui = mu[clusters[celli]]
            muabg = torch.log(mui + torch.sum(abg))

            
            if debug:
                print('celli')
                print(celli)
                print('whichi')
                print(whichi)
                print('ai')
                print(ai)
                print('bi')
                print(bi)
                print('ti')
                print(ti)
                print('yi')
                print(yi)
                print('neg_bt')
                print(neg_bt)
                print('a_adj')
                print(a_adj)
                print('abg')
                print(abg)
                print('mui')
                print(mui)
                print('muabg')
                print(muabg)
                
            logL += muabg

        logL -= torch.sum(mu[clusters])*self.tmax

        G = torch.tensor(np.zeros((N,nc)))
        Dt = torch.tensor(self.tmax - y_sorted)
        #print(Dt)
        #print(torch.log(Dt))
        for i in range(N):
            #gb = logN.cdf(Dt)
            #print(gb.grad)
            gb = log_normal_cdf(Dt, b[clusters[i],clusters[cells_sorted]])
            #gb = torch.div(torch.log(Dt),b[clusters[i],clusters[cells_sorted]])
            
            
            ga = torch.mul(a[clusters[i],clusters[cells_sorted]], adj[i,cells_sorted])
            G[i] = torch.mul(ga,gb)
        
        if debug:
            print('G')
            print(G)
            print(' ')
        logL -= torch.sum(torch.sum(G))
        
        if a_prior:
            #logL += torch.sum((alpha-1) * torch.log(a) + (beta - 1) * torch.log(1 - a))
            beta = torch.sum(torch.distributions.Beta(alpha, beta).log_prob(a))
            logL += beta
        

        return -logL
    
    
    def set_init(self, mu_init = None, a_init = None, ajj_init = None, 
                 b_init = None, bjj_init = None, epsilon_init = None, 
                 nclusters = None, tmax = None):
        
        self.format_data()
        if nclusters is None:
                nclusters = self.nclusters
        mu_other = len(self.y_sorted)/(self.N * np.max(self.y_sorted))
        b_est = self.est_b_init(self.peaks)
        
        if mu_init is None:
            self.mu_pre = torch.tensor(softplus_inv(0.5*mu_other*np.ones(nclusters)), requires_grad = True)
        else:
            self.mu_pre = torch.tensor(softplus_inv(mu_init), requires_grad = True)
        
        if a_init is None:    
            self.a_pre = torch.tensor(softplus_inv(0.5*mu_other*np.ones((nclusters,nclusters))), requires_grad = True)
        else:
            self.a_pre = torch.tensor(softplus_inv(a_init), requires_grad = True) 
            
        if ajj_init is None:    
            self.ajj_pre = torch.tensor(np.ones(nclusters), requires_grad = True)
        else:
            self.ajj_pre = torch.tensor(softplus_inv(ajj_init), requires_grad = True)
                                        
        if b_init is None:
            self.b_pre = torch.tensor(softplus_inv(np.ones((nclusters,nclusters))), 
                                      requires_grad = True)
        else:
            self.b_pre = torch.tensor(softplus_inv(b_init), requires_grad = True)
            
        if bjj_init is None:
            self.bjj_pre = torch.tensor(softplus_inv(np.ones((nclusters,))), 
                                      requires_grad = True)
        else:
            self.bjj_pre = torch.tensor(softplus_inv(bjj_init), requires_grad = True)

        if epsilon_init is None:
            self.epsilon_pre = torch.tensor(10., requires_grad = True)
        else:
            self.epsilon_pre = torch.tensor(softplus_inv(epsilon_init), requires_grad = True)
            
        if tmax is None:
            if self.tmax is not None:
                pass
            else:
                print('Please enter maximum time')
        else:
            self.tmax = tmax
            
        return
        
            
        
    def fit(self, iters = 1000, sharpness = 1., lr = 1e-4, a_prior = False, kernel = None, control = False):
        self.sharpness = sharpness
        self.converged = False
        y_sorted, cells_sorted, d = self.format_data()
        clusters = self.clusters
        optimizer = torch.optim.SGD([self.mu_pre, self.a_pre, self.b_pre], lr = lr)
        #optimizer = torch.optim.SGD([self.a_pre], lr = lr)
        print_step = int(iters/10)
        if print_step == 0:
            print_step = 1
        loss_old = self.negLogL(y_sorted, cells_sorted, d, clusters, a_prior = a_prior, kernel = kernel, control = control)
        for i in range(iters):
            optimizer.zero_grad()
            loss = self.negLogL(y_sorted, cells_sorted, d, clusters, a_prior = a_prior, kernel = kernel, control = control)
            loss.backward()
            
#             print(self.mu_pre.grad)
#             print(self.a_pre.grad)
#             print(self.b_pre.grad)
            
            optimizer.step()

            if i % print_step == 0:
                print(str(i) + '/' + str(iters) + ' logL: ' + str(loss.item()))
                        
            if loss_old < loss:
                self.converged = True
                print('converged <')
                break
            elif (np.abs((loss_old.item() - loss.item())/loss_old.item()) < .000001) & (i > 1000): #was .00001 and 500
                self.converged = True
                print('converged %')
                break
            loss_old = loss

            if (loss != loss).item():
                print('NaNs encountered')
                break
                    
        print('Final logL: ' + str(loss.item()) + ' at iteration ' + str(i))
        self.loss = loss.item()
        return
    
    def get_a_learned(self):
        #return softplus(self.a_pre.data)
        return softplus(self.a_pre.data) #1./(1 + np.exp(-self.a_pre.data))
                                        
    def get_ajj_learned(self):
        return softplus(self.ajj_pre.data)
    
    def get_b_learned(self):
        return softplus(self.b_pre.data)
    
    def get_bjj_learned(self):
        return softplus(self.bjj_pre.data)
    
    def get_mu_learned(self):
        return softplus(self.mu_pre.data)
    
    def get_epsilon_learned(self):
        return softplus(self.epsilon_pre.data)
    
    
    def animate(self, length = 0.01, amplitude = 1,
                smin = 5, smax = 80):
        
        def lin_transform(x):
            return smax * (x - obsmin)/(obsmax - obsmin) + smin
        
        peaks = self.peaks
        N = len(peaks)
        t = np.arange(np.min(np.hstack(peaks)),np.max(np.hstack(peaks)) + 1,0.05)
        nt = len(t)
        obs = np.zeros((nt,N))
        for i in range(N):
            peaks_i = peaks[i]
            npeaks = len(peaks_i)
            t_repeats = np.tile(np.expand_dims(t,1), (1,npeaks))
            dt_peaks = t_repeats - peaks_i
            each_peak_obs = amplitude * np.exp(-np.square(dt_peaks/length))
            total_obs = np.sum(each_peak_obs,axis = 1)
            obs[:,i] = total_obs
        obsmin = np.min(obs)
        obsmax = np.max(obs)
        fig2 = plt.figure()

        x = self.coords[:,0]
        y = self.coords[:,1]
        #base = np.hypot(x, y)
        ims = []
        for add in np.arange(nt):
            ims.append((plt.scatter(x, y, c =  self.clusters, s = lin_transform(obs[add]), edgecolor = 'k'),))

        im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                           blit=True)
        anim = animation.ArtistAnimation(fig2, ims, interval=100, repeat_delay=3000,
                                           blit=True)
        #HTML(anim.to_jshtml())
        return anim
    
    
   
    
    def est_b_init(self, peaks):
        count = 0
        total = 0
        for i in range(len(peaks)):
            if len(peaks[i]) > 1:
                count += len(peaks[i])
                total += np.sum(np.diff(peaks[i]))
            else:
                pass
        if count == 0:
            count = 1.
            total = 1.
        mean = np.divide(total, count)
        if np.log(mean) < 0:
            sig_est = np.sqrt(-np.log(mean))
        else:
            sig_est = np.sqrt(2 * np.log(mean))
        return sig_est
    
    def simulate_with_self(self, N = 100, 
                           tmax = 16,
                           peakmax = 5000,
                           nclusters = 3, 
                           p = None,
                           xmax = 700, 
                           mu = None,
                           a = None, ajj = None, 
                           b = None, bjj = None,
                           epsilon =  None, debug = False):
        
        if mu is None:
            #mu = 10.** np.random.uniform(-2,0.3, size = nclusters)
            mu = np.random.uniform(0, 2, size = nclusters)
        
        if a is None:
            a = np.random.uniform(0, 2, size = (nclusters,nclusters))
            
            
        if b is None:
            b = np.random.uniform(0, 4, size = (nclusters,nclusters))
            #np.multiply(10.**np.random.uniform(0,1,size = (nclusters, nclusters)), mu) # was -0.5 to 0.5 in test8
        
        if ajj is None:
            ajj = np.random.uniform(0, 2, size = (nclusters,))
            
            
        if bjj is None:
            bjj = np.random.uniform(0, 4, size = (nclusters,))
        
        if epsilon is None:
            epsilon = np.random.uniform(25,100)
            
        if p is None:
            p = np.random.dirichlet(np.ones(nclusters))
            
            
        coords = np.random.uniform(0,xmax,size=(N,2))
        clusters = np.random.choice(nclusters, size = N, p = p)
        d = squareform(pdist(coords))
        
        lambda_0 = mu[clusters] # array of size n cells with mu for each cell
        tnow = 0
        tlast = np.zeros(N) # intialize the last peak for each cell
        ti_all = np.random.exponential(1./lambda_0) #drawn the next peak for each cell
        if debug:
            print('ti all')
            print(ti_all)
        
        peaks = [[]] * N # initialize array of array listing peaks
        
        ajj_diag = np.diag(ajj[clusters] - 1) + np.ones((N,N))
        adj = np.multiply((d < epsilon), ajj_diag) # define the cell-cell interactions
        
        if debug:
            print('adj')
            print(adj.shape)
            print(adj)
        
        bjj_adj = np.ones((N,N)) + np.diag(bjj[clusters] - 1.)
        
        if debug:
            print('bjj_adj')
            print(bjj_adj.shape)
            print(bjj_adj)
        
        tnow = np.min(ti_all) # move to next peak
        cellnow = np.argmin(ti_all)
        peaks[cellnow] = np.append(peaks[cellnow], tnow)
        ts = np.expand_dims(np.array(tnow), axis = 0) # array formatting
        
        if debug:
            print('tnow')
            print(tnow)
            print('cellnow')
            print(cellnow)
            print('peaks')
            print(peaks)
            
            print(' ')
        
        which_cell = np.expand_dims(np.array([np.argmin(ti_all)]), axis = 1) # move to cell
        
        
        
        ## calculate conditional intensity for each cell
        #a_add = np.multiply(a[clusters[which_cell[-1]],clusters], 
        #                    adj[which_cell[-1]]) # ajk for each cell, k is last cell        
        
        a_add = np.multiply(a[clusters[which_cell[-1]],clusters], adj[which_cell[-1]])
        if debug:
            print('a_add')
            print(a_add)
            
        b_add = np.multiply(b[clusters[which_cell[-1]],clusters], bjj_adj[which_cell[-1]])
        
        if debug:
            print('b_add')
            print(b_add)
        
        neg_bt = np.reshape(lognorm.pdf(tnow, b_add), -1)
        if debug:
            print('neg_bt')
            print(neg_bt)
            
        kernel = np.multiply(a_add, neg_bt) + lambda_0
        if debug:
            print('kernel')
            print(kernel)
        ti_all = np.random.exponential(1./kernel)
        if debug:
            print('ti_all')
            print(ti_all)
            print(' ')
        
        while tnow < tmax:
            tnow = tnow + np.min(ti_all)

                
            
            cellnow = np.argmin(ti_all)
            #print('cellnow')
            #print(cellnow)
            ts = np.append(ts, tnow)
            ts = np.expand_dims(ts, axis = 0)
            which_cell = np.append(which_cell, cellnow)
            peaks[cellnow] = np.append(peaks[cellnow], tnow)

            ts_calc = np.repeat(ts, N, axis = 0)

            kernel = np.zeros(N)
            if debug:
                print('tnow')
                print(tnow)
                print('cellnow')
                print(cellnow)
                print('peaks')
                print(peaks)
            for i in range(len(which_cell)):
                a_add = np.multiply(a[clusters[which_cell[i]],clusters], adj[which_cell[i]])

                b_add = np.multiply(b[clusters[which_cell[i]],clusters], bjj_adj[which_cell[i]])

                neg_bt = np.reshape(lognorm.pdf(tnow - ts[0,i], b_add), -1)

                kernel += np.multiply(a_add, neg_bt)

                if debug:
                    print('ti')
                    print(ts[0,i])
                    print('cell i')
                    print(which_cell[i])
                    print('a_add')
                    print(a_add)
                    print('b_add')
                    print(b_add)
                    print('negbt')
                    print(neg_bt)
                    print('kernel')
                    print(kernel)
                    print(' ')
                    
            lambda_i = lambda_0 + kernel
            if debug:
                print('lambda_i')
                print(lambda_i)
            ti_all = np.random.exponential(1./lambda_i)
            if debug:
                print('ti_all')
                print(ti_all)
                print(' ')
                print(' ')
            if(len(ts[0])) > peakmax:
                print(len(ts[0])-1)
                break
        tnow = tnow + np.min(ti_all)  # one last draw to set tmax (length of time with no peaks observed)  
        self.peaks = peaks
        self.d = d
        self.coords = coords
        self.clusters = clusters
        self.a_true = a
        self.ajj_true = ajj
        self.b_true = b
        self.bjj_true = bjj
        self.mu_true = mu
        self.ep_true = epsilon
        self.nclusters = nclusters
        self.N = N
        self.tmax = tnow
        
        return
    
    
    def negLogL_with_self(self, y_sorted, cells_sorted, d, clusters, 
                debug = False, a_prior = False, alpha = .001, beta = .001,
                kernel = None, control = False):
        
        
        mu = torch.log(1 + torch.exp(self.mu_pre))
        a = torch.log(1 + torch.exp(self.a_pre))
        ajj = torch.log(1 + torch.exp(self.ajj_pre))
        b = torch.log(1 + torch.exp(self.b_pre))
        bjj = torch.log(1 + torch.exp(self.bjj_pre))
        epsilon = torch.log(1 + torch.exp(self.epsilon_pre))


        nc = len(y_sorted)
        N = d.shape[0]
        D = torch.tensor(d)
        
        
        
        if kernel is None or kernel == 'logistic':
            sharpness = self.sharpness
            adj = torch.div(torch.exp(-sharpness*(D - epsilon)), 1. + torch.exp(-sharpness*(D - epsilon)))
        elif kernel == 'relu':
            adj = (1./epsilon) * torch.nn.functional.relu(epsilon - D)
        
        elif kernel == 'exp':
            adj = torch.exp(-D/epsilon)
            
        elif kernel == 'fixed':
            adj = D < epsilon
            
        else:
            adj = torch.div(torch.exp(-sharpness*(D - epsilon)), 
                                                    1. + torch.exp(-sharpness*(D - epsilon)))
            
        if control:
            adj = torch.diag(torch.ones(N))
        
        #print(adj)
        intermediate = torch.diag(ajj[clusters] - 1.) + torch.ones((N,N))
        adj = torch.mul(intermediate, adj)
        
        #print(adj)
        
        bjj_adj = torch.diag(bjj[clusters] - 1.) + torch.ones((N,N))
        #tmax = y_sorted[-1]
        
        #print(bjj_adj)

        logL = torch.tensor(0.)
        
        
        #logN = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        #logN = LogNormal(torch.tensor([0.0]), b)
        for i in range(1,nc):
            celli = cells_sorted[i].astype(np.int)
            #print(celli)
            
            whichi = cells_sorted[0:i].astype(np.int)
            #print(whichi)
            
            ai = a[clusters[celli],clusters[whichi]]
            #print('ai')
            #print(ai)
            
            bi = b[clusters[celli],clusters[whichi]]
            b_adj = torch.mul(bi, bjj_adj[celli, cells_sorted[0:i]])
            #print(b_adj)
            #print(' ')
            
            
            ti = y_sorted[0:i]
            yi = torch.tensor(y_sorted[i] - ti)
            
            neg_bt = log_normal_pdf(yi,b_adj)
            a_adj = torch.mul(ai, adj[celli, cells_sorted[0:i]])
            #print('a')
            #print(a_adj)
            #print(' ')
            
            abg = torch.mul(a_adj, neg_bt)
            mui = mu[clusters[celli]]
            muabg = torch.log(mui + torch.sum(abg))

            
            if debug:
                print('celli')
                print(celli)
                print('whichi')
                print(whichi)
                print('ai')
                print(ai)
                print('bi')
                print(bi)
                print('ti')
                print(ti)
                print('yi')
                print(yi)
                print('neg_bt')
                print(neg_bt)
                print('a_adj')
                print(a_adj)
                print('b_adj')
                print(b_adj)
                print('abg')
                print(abg)
                print('mui')
                print(mui)
                print('muabg')
                print(muabg)
                print(' ')
                
            logL += muabg

        logL -= torch.sum(mu[clusters])*self.tmax
        if debug:
            print('mu[clusters]')
            print(mu[clusters])

        G = torch.tensor(np.zeros((N,nc)))
        Dt = torch.tensor(self.tmax - y_sorted)
        #print(Dt)
        #print(torch.log(Dt))
        for i in range(N):
            #gb = logN.cdf(Dt)
            #print(gb.grad)
            gbb = torch.mul(b[clusters[i],clusters[cells_sorted]], bjj_adj[i,cells_sorted])
            gb = log_normal_cdf(Dt, gbb)
            #gb = torch.div(torch.log(Dt),b[clusters[i],clusters[cells_sorted]])
            
            
            ga = torch.mul(a[clusters[i],clusters[cells_sorted]], adj[i,cells_sorted])
            G[i] = torch.mul(ga,gb)
        
            if debug:
                print('Dt')
                print(Dt)
                print('gbb')
                print(gbb)
                print('gb')
                print(gb)
                print('ga')
                print(ga)
                print('G')
                print(G[i])
                print(' ')
        logL -= torch.sum(torch.sum(G))
        
        
        return -logL
    
    
    def fit_with_self(self, iters = 1000, sharpness = 1., lr = 1e-4, a_prior = False, kernel = None):
        self.sharpness = sharpness
        self.converged = False
        y_sorted, cells_sorted, d = self.format_data()
        clusters = self.clusters
        optimizer = torch.optim.SGD([self.mu_pre, 
                                     self.a_pre, self.ajj_pre,
                                     self.b_pre, self.bjj_pre], lr = lr)
        print_step = int(iters/10)
        if print_step == 0:
            print_step = 1
        loss_old = self.negLogL_with_self(y_sorted, cells_sorted, d, clusters, a_prior = a_prior, kernel = kernel)
        for i in range(iters):
            optimizer.zero_grad()
            loss = self.negLogL_with_self(y_sorted, cells_sorted, d, clusters, a_prior = a_prior, kernel = kernel)
            loss.backward()
            
#             print(self.mu_pre.grad)
#             print(self.a_pre.grad)
#             print(self.b_pre.grad)
            
            optimizer.step()

            if i % print_step == 0:
                print(str(i) + '/' + str(iters) + ' logL: ' + str(loss.item()))
                        
            if loss_old < loss:
                self.converged = True
                print('converged <')
                break
            elif (np.abs((loss_old.item() - loss.item())/loss_old.item()) < .00001) & (i > 500):
                self.converged = True
                print('converged %')
                break
            loss_old = loss

            if (loss != loss).item():
                print('NaNs encountered')
                break
                    
        print('Final logL: ' + str(loss.item()) + ' at iteration ' + str(i))
        self.loss = loss.item()
        return
    
    
    def lambda_i(self, y_sorted, cells_sorted, d, clusters, 
                 mu, a, ajj, b, bjj, epsilon, peak,
                 debug = False, a_prior = False, alpha = .001, beta = .001,
                 kernel = 'fixed', control = False):
        
        nc = len(y_sorted)
        N = d.shape[0]
        D = torch.tensor(d)
        
        
        
        if kernel is None or kernel == 'logistic':
            sharpness = self.sharpness
            adj = torch.div(torch.exp(-sharpness*(D - epsilon)), 1. + torch.exp(-sharpness*(D - epsilon)))
        elif kernel == 'relu':
            adj = (1./epsilon) * torch.nn.functional.relu(epsilon - D)
        
        elif kernel == 'exp':
            adj = torch.exp(-D/epsilon)
            
        elif kernel == 'fixed':
            adj = D < epsilon
            
        else:
            adj = torch.div(torch.exp(-sharpness*(D - epsilon)), 
                                                    1. + torch.exp(-sharpness*(D - epsilon)))
            
        if control:
            adj = torch.diag(torch.ones(N))
        
        #print(adj)
        #intermediate = torch.diag(ajj[clusters] - 1.) + torch.ones((N,N))
        #adj = torch.mul(intermediate, adj)
        
        #print(adj)
        
        #bjj_adj = torch.diag(bjj[clusters] - 1.) + torch.ones((N,N))
        #tmax = y_sorted[-1]
        
        #print(bjj_adj)

        logL = torch.tensor(0.)
        
        
        #logN = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        #logN = LogNormal(torch.tensor([0.0]), b)
        for i in range(peak,peak+1):
            celli = cells_sorted[i].astype(np.int)
            #print(celli)
            
            whichi = cells_sorted[0:i].astype(np.int)
            #print(whichi)
            
            ai = a[clusters[celli],clusters[whichi]]
            ai[whichi == celli] = ajj[clusters[celli]]
            #print('ai')
            #print(ai)
            
            bi = b[clusters[celli],clusters[whichi]]
            bi[whichi == celli] = bjj[clusters[celli]]
            b_adj = bi #torch.mul(bi, bjj_adj[celli, cells_sorted[0:i]])
            #print(b_adj)
            #print(' ')
            
            
            ti = y_sorted[0:i]
            yi = torch.tensor(y_sorted[i] - ti)
            
            neg_bt = log_normal_pdf(yi,b_adj)
            a_adj = torch.mul(ai, adj[celli, cells_sorted[0:i]])
            #print('a')
            #print(a_adj)
            #print(' ')
            
            abg = torch.mul(a_adj, neg_bt)
            mui = mu[clusters[celli]]
            muabg = mui + torch.sum(abg)

            
            if debug:
                print('celli')
                print(celli)
                print('whichi')
                print(whichi)
                print('ai')
                print(ai)
                print('bi')
                print(bi)
                print('ti')
                print(ti)
                print('yi')
                print(yi)
                print('neg_bt')
                print(neg_bt)
                print('a_adj')
                print(a_adj)
                print('b_adj')
                print(b_adj)
                print('abg')
                print(abg)
                print('mui')
                print(mui)
                print('muabg')
                print(muabg)
                print(' ')
                
        
        if debug:
            print('mu[clusters]')
            print(mu[clusters])
        
        return muabg
    
    
    def get_fisher_information(self, mu_hat = None, a_hat = None, ajj_hat = None, b_hat = None, bjj_hat = None, 
                               epsilon = None):
        
        if mu_hat is None:
            mu_hat = self.get_mu_learned().item()
            mu_hat = np.multiply(mu_hat, np.ones((self.nclusters,)))
        if a_hat is None:
            a_hat = self.get_a_learned().item()
            a_hat = np.multiply(a_hat, np.ones((self.nclusters,self.nclusters)))
        if ajj_hat is None:
            ajj_hat = self.get_ajj_learned().item()
            ajj_hat = np.multiply(np.multiply(ajj_hat, a_hat), np.ones((self.nclusters,)))
        if b_hat is None:
            b_hat = self.get_b_learned().item()
            b_hat = np.multiply(b_hat, np.ones((self.nclusters,self.nclusters)))
        if bjj_hat is None:
            bjj_hat = self.get_bjj_learned().item()
            bjj_hat = np.multiply(np.multiply(bjj_hat, b_hat), np.ones((self.nclusters,)))
        if epsilon is None:
            epsilon = self.get_epsilon_true()
        else:
            epsilon = torch.tensor(epsilon)
            

        npeaks = len(self.y_sorted)
        
        Sigma = np.zeros((5,5))
        
        for i in range(npeaks):
            
            mu = torch.tensor(mu_hat, requires_grad = True)
            a = torch.tensor(a_hat, requires_grad = True)
            b = torch.tensor(b_hat, requires_grad = True)
            ajj = torch.tensor(ajj_hat, requires_grad = True)
            bjj = torch.tensor(bjj_hat, requires_grad = True)
        
            li = self.lambda_i(self.y_sorted, self.cells_sorted, self.d, self.clusters, 
                               mu, a, ajj, b, bjj, epsilon, i)
            optimizer = torch.optim.SGD([mu, a, ajj, b, bjj], lr = 1e-4)
            optimizer.zero_grad()
            li.backward()
            dl_dmu = mu.grad.item()
            dl_da = a.grad.item()
            #print('a : ' , end = ' ' )
            #print(dl_da)
            dl_dajj = ajj.grad.item()
            dl_db = b.grad.item()
            #print('b : ' , end = ' ')
            #print(dl_db)
            dl_dbjj = bjj.grad.item()
            #print('bjj : ' , end = ' ')
            #print(dl_dbjj)
            
            #print('li : ' + str(li.item()))
            
            dl_all = np.array([dl_dmu, dl_da, dl_dajj, dl_db, dl_dbjj])
            Sigma += np.outer(dl_all, dl_all)/li.item()
            #print(Sigma[1,1])
            #print(Sigma[3,3])
            
        return Sigma
    
    def get_confidence_intervals(self, mu_hat = None, a_hat = None, ajj_hat = None, b_hat = None, bjj_hat = None, 
                                  epsilon = None):
        
        asymptotic_covariance = np.linalg.inv(self.get_fisher_information(mu_hat, 
                                                                             a_hat, 
                                                                             ajj_hat, 
                                                                             b_hat, 
                                                                             bjj_hat, 
                                                                             epsilon))
        confidence_intervals = 1.96 * np.sqrt(np.diag(asymptotic_covariance))
        return confidence_intervals