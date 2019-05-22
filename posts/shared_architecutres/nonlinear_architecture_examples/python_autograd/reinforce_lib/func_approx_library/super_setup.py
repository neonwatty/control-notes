import autograd.numpy as np
from . import super_optimizers 
from . import super_cost_functions
from . import multilayer_perceptron
 
class Setup:
    def __init__(self):
        # make containers for all histories
        self.weight_history = []
        self.cost_history = []
         
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # create training and testing cost functions
        self.cost_object = super_cost_functions.Setup(name,**kwargs)
        self.cost_name = name
     
    #### define feature transformation ####
    def choose_features(self,layer_sizes,**kwargs):         
        # multilayer perceptron #
        transformer = multilayer_perceptron.Setup(layer_sizes,**kwargs)
        self.feature_transforms = transformer.feature_transforms
        self.weight_initializer = transformer.initializer
        self.layer_sizes = transformer.layer_sizes
                      
        ### with feature transformation constructed, pass on to cost function ###
        self.cost_object.define_feature_transform(self.feature_transforms)
        self.cost = self.cost_object.cost
        self.model = self.cost_object.model
             
    #### run optimization ####
    def fit(self,x,y,max_its,alpha,**kwargs):
        # set initialization
        self.w_init = self.weight_initializer()
        if len(self.weight_history) > 0:
            self.w_init = self.weight_history[-1]
        else:
            self.weight_history.append(self.w_init)
         
        # batch size for gradient descent?
        self.batch_size = x.shape[1]
        if 'batch_size' in kwargs:
            self.batch_size = min(kwargs['batch_size'],self.batch_size)
         
        # verbose or not
        verbose = True
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
 
        ## run algo ##
        w_hist = []
        cost_hist = []
        algo = kwargs['algo']
        
        # run gradient descent
        if algo == 'sgd':
            w_hist,cost_hist = super_optimizers.gradient_descent(self.cost,self.w_init,x,y,alpha,max_its,self.batch_size,verbose=verbose)
            
        # run RMSprop
        if algo == 'RMSprop':
            # if steps have been taken previously, use compuated average
            if len(self.weight_history) == 1:
                self.avg_sq_grad = np.ones(np.size(self.weight_history[-1]))
            w_hist,cost_hist,self.avg_sq_grad = super_optimizers.RMSprop(self.cost,self.w_init,x,y,alpha,max_its,self.batch_size,verbose=verbose,avg_sq_grad=self.avg_sq_grad)
                                                                                          
        # store all new histories
        for j in range(1,len(w_hist)):
            w = w_hist[j]
            t = cost_hist[j]
            self.weight_history.append(w)
            self.cost_history.append(t)
    
    # construct predictor
    def predict(self,value):
        w = self.weight_history[-1]
        return self.model(value,w)