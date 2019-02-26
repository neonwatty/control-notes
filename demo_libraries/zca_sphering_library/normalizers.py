import numpy as np
import copy
import math
import time

class Tools:

    # center input data
    def center(self,x):
        x_means = np.mean(x,axis=0)[np.newaxis,:]
        x_centered = x - x_means
        return x_centered
        
    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(x_stds < 10**(-5))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((x_stds.shape))
            adjust[ind] = 1.0
            x_stds += adjust

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        # return normalizer 
        return normalizer,inverse_normalizer

    # use standard normalizer to normalize input
    def standard_normalize(self,x):
        # create functions to sphere and un-sphere data
        forward_func, inverse_func = self.standard_normalizer(x)
        
        # sphere data
        x_spherd = forward_func(x)
        return x_spherd
    
    # A contrast-normalizing function
    def contrast_normalize(self,x):
        # contrast normalize
        forward_func, inverse_func = self.standard_normalizer(x)
        x_normalized = forward_func(x)
        return x_normalized

    # sphereing pre-processing functionality 
    def PCA(self,x,**kwargs):
        # regularization parameter for numerical stability
        lam = 10**(-7)
        if 'lam' in kwargs:
            lam = kwargs['lam']

        # create the correlation matrix
        P = float(x.shape[1])
        Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

        # use numpy function to compute eigenvalues / vectors of correlation matrix
        d,V = np.linalg.eigh(Cov)
        return d,V

    # PCA-sphere - use PCA to normalize input features
    def PCA_spherer(self,x,**kwargs):
        # Step 1: mean-center the data
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_centered = x - x_means

        # Step 2: compute pca transform on mean-centered data
        d,V = self.PCA(x_centered,**kwargs)

        # Step 3: divide off standard deviation of each (transformed) input, 
        # which are equal to the returned eigenvalues in 'd'.  
        stds = (d[:,np.newaxis])**(0.5)
        
        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((stds.shape))
            adjust[ind] = 1.0
            stds += adjust
        
        # create normalizer / inverse-normalizer
        normalizer = lambda data: np.dot(V.T,data - x_means)/stds

        # create inverse normalizer
        inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

        # return normalizer 
        return normalizer,inverse_normalizer
    
    # use PCA spherer to normalize input data
    def PCA_sphere(self,x):
        # create functions to sphere and un-sphere data
        forward_func, inverse_func = self.PCA_spherer(x)
        
        # sphere data
        x_spherd = forward_func(x)
        return x_spherd
    
    # ZCA spherer 
    def ZCA_spherer(self,x,**kwargs):
        # Step 1: mean-center the data
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_centered = x - x_means
        
        # Step 2: compute pca transform on mean-centered data
        d,V = self.PCA(x_centered,**kwargs)

        # Step 3: divide off standard deviation of each (transformed) input, 
        # which are equal to the returned eigenvalues in 'd'.  
        stds = (d[:,np.newaxis])**(0.5)
        
        # check to make sure thta x_stds > small threshold, for those not
        # divide by 1 instead of original standard deviation
        ind = np.argwhere(stds < 10**(-2))
        if len(ind) > 0:
            ind = [v[0] for v in ind]
            adjust = np.zeros((stds.shape))
            adjust[ind] = 1.0
            stds += adjust
        
        # create normalizer / inverse-normalizer
        normalizer = lambda data: np.dot(V,np.dot(V.T,data - x_means)/stds)

        # create inverse normalizer
        inverse_normalizer = lambda data: np.dot(V.T,np.dot(V,data*stds) + x_means)

        # return normalizer 
        return normalizer,inverse_normalizer        
        
    # ZCA-sphereing - use ZCA to normalize input features
    def ZCA_sphere(self,x):
        # create functions to sphere and un-sphere data
        forward_func, inverse_func = self.ZCA_spherer(x)
        
        # sphere data
        x_spherd = forward_func(x)
        return x_spherd