import autograd.numpy as np
from inspect import signature

class Setup:
    def __init__(self,name,feature_transforms,**kwargs):       
        # make copy of feature transformation
        self.feature_transforms = feature_transforms
        
        # for two-class classification
        if name == 'softmax':
            self.cost = self.softmax
        if name == 'perceptron':
            self.cost = self.perceptron
        if name == 'twoclass_counter':
            self.cost = self.counter
        if name == 'twoclass_accuracy':
            self.cost = self.twoclass_accuracy
            
        # for multiclass classification
        if name == 'multiclass_perceptron':
            self.cost = self.multiclass_perceptron
        if name == 'multiclass_softmax':
            self.cost = self.multiclass_softmax
        if name == 'multiclass_counter':
            self.cost = self.multiclass_counter
        if name == 'multiclass_accuracy':
            self.cost = self.multiclass_accuracy
        
            
    ###### cost functions #####
    # an implementation of our model employing a nonlinear feature transformation
    def model(self,x,w):    
        # feature transformation 
        f = self.feature_transforms(x,w[0])

        # compute linear combination and return
        a = w[1][0] + np.dot(f.T,w[1][1:])
        return a.T
    
    ###### two-class classification costs #######
    # the convex softmax cost function
    def softmax(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.log(1 + np.exp(-y_p*self.model(x_p,w))))
        return cost/float(np.size(y_p))

    # the convex relu cost function
    def relu(self,w,x,y,iter):
        # get batch of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # compute cost over batch
        cost = np.sum(np.maximum(0,-y_p*self.model(x_p,w)))
        return cost/float(np.size(y_p))
    
    # the counting cost function
    def twoclass_counter(self,w,x,y):
        # compute predicted labels
        y_hat = np.sign(model(x,w))

        # count misclassifications
        count = len(np.argwhere(y != y_hat))
        return count
    
    # twoclass accuracy function
    def twoclass_accuracy(self,w,x,y):
        # compute number of misclassifications
        count = self.twoclass_counting_cost(w,x,y)

        # compute accuracy and return
        acc = 1 - (count/y.size)
        return acc

    ###### multiclass classification costs #######
    # multiclass perceptron
    def multiclass_perceptron(self,w,x,y,iter):
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]

        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute maximum across data points
        a =  np.max(all_evals,axis = 0)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(y_p))

    # multiclass softmax
    def multiclass_softmax(self,w,x,y,iter):   
        # get subset of points
        x_p = x[:,iter]
        y_p = y[:,iter]
        
        # pre-compute predictions on all points
        all_evals = self.model(x_p,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 0)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
        cost = np.sum(a - b)

        # return average
        return cost/float(np.size(y_p))

    # multiclass misclassification cost function - aka the fusion rule
    def multiclass_counter(self,w,x,y):                
        # pre-compute predictions on all points
        all_evals = self.model(x,w)

        # compute predictions of each input point
        y_hat = (np.argmax(all_evals,axis = 0))[np.newaxis,:]
        count = len(np.argwhere(y_hat != y))

        # return number of misclassifications
        return count
    
    # multiclass accuracy
    def multiclass_accuracy(self,w,x,y):        
        # compute number of misclassifications
        count = self.multiclass_counter(w,x,y)
        
        # compute accuracy and return
        acc = 1 - (count/y.size)
        return acc