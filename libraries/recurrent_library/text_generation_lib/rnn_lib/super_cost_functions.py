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
        if name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
        if name == 'multiclass_counter':
            self.cost = self.multiclass_counting_cost
            
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
    
    # multiclass softmax
    def multiclass_softmax(self,w,x,y):        
        # pre-compute predictions on all points
        all_evals = self.model(x,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 0)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(y))

    # multiclass misclassification cost function - aka the fusion rule
    def multiclass_counting_cost(self,w,x,y):                
        # pre-compute predictions on all points
        all_evals = self.model(x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 0))[np.newaxis,:]

        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(y - y_predict)))

        # return number of misclassifications
        return count