import autograd.numpy as np
from inspect import signature

class Setup:
    def __init__(self,name,model,**kwargs):        
        ### make cost function choice ###
        # for regression
        if name == 'least_squares':
            self.cost = self.least_squares
        if name == 'least_absolute_deviations':
            self.cost = self.least_absolute_deviations
            
        # define model
        self.model = model

    ###### regression costs #######    
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w,x,y):
        # compute cost over batch
        cost = np.sum((self.model(x,w) - y)**2)
        return cost/float(np.size(y))

    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w,x,y):
        # compute cost over batch
        cost = np.sum(np.abs(self.model(x,w) - y))
        return cost/float(np.size(y))
