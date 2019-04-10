import autograd.numpy as np
from . import super_optimizers 
from . import super_cost_functions
from . import normalizers
from . import history_plotters
import copy

class Setup:
    def __init__(self,x,y,**kwargs):
        # link in data
        self.x = x
        self.y = y
        
        # make containers for all histories
        self.weight_histories = []
        self.train_cost_histories = []
        self.train_accuracy_histories = []
        self.val_cost_histories = []
        self.val_accuracy_histories = []
        self.train_costs = []
        self.val_costs = []
        
    #### define preprocessing steps ####
    def preprocessing_steps(self,**kwargs):        
        ### produce / use data normalizer ###
        normalizer_name = 'standard'
        if 'normalizer_name' in kwargs:
            normalizer_name = kwargs['normalizer_name']
        self.normalizer_name = normalizer_name

        # produce normalizer / inverse normalizer
        s = normalizers.Setup(self.x,normalizer_name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x = self.normalizer(self.x)
       
    #### split data into training and validation sets ####
    def make_train_val_split(self,train_portion):
        # translate desired training portion into exact indecies
        self.train_portion = train_portion
        r = np.arange(self.x.shape[1])
        train_num = int(np.round(train_portion*len(r)))
        self.train_inds = r[:train_num]
        self.val_inds = r[train_num:]
        
        # define training and testing sets
        self.x_train = self.x[:,self.train_inds]
        self.x_val = self.x[:,self.val_inds]
        
        self.y_train = self.y[:,self.train_inds]
        self.y_val = self.y[:,self.val_inds]
     
    #### define cost function ####
    def choose_cost(self,name,model,**kwargs):
        # create training and testing cost functions
        self.cost_object = super_cost_functions.Setup(name,model,**kwargs)
        self.cost_name = name
    
        ### with feature transformation constructed, pass on to cost function ###
        self.cost = self.cost_object.cost
        self.model = self.cost_object.model
        
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            self.count_object = super_cost_functions.Setup('twoclass_counter',model,**kwargs)
                        
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            self.count_object = super_cost_functions.Setup('multiclass_counter',model,**kwargs)

    #### run optimization ####
    def fit(self,**kwargs):
        # basic parameters for gradient descent run (default algorithm)
        max_its = 500; alpha_choice = 10**(-1);
        
        # set parameters by hand
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            self.alpha_choice = kwargs['alpha_choice']
        
        optimizer = 'gradient_descent'
        if 'optimizer' in kwargs:
            optimizer = kwargs['optimizer']
            
        # verbose or not
        verbose = True
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            
        self.batch_size = np.size(self.y_train)
        if 'batch_size' in kwargs:
            self.batch_size = min(kwargs['batch_size'],self.batch_size)
            
        # set initialization
        self.w_init = kwargs['w_init']

        # optimize
        weight_history = []
        train_cost_history = []
        val_cost_history = []

       # run gradient descent
        weight_history,train_cost_history,val_cost_history = super_optimizers.gradient_descent(self.cost,self.w_init,self.x_train,self.y_train,self.x_val,self.y_val,self.alpha_choice,self.max_its,self.batch_size,verbose=verbose)
                                                                                         
        # store all new histories
        self.weight_histories.append(weight_history)
        self.train_cost_histories.append(train_cost_history)
        self.val_cost_histories.append(val_cost_history)

        # if classification produce count history
        if self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            train_accuracy_history = [1 - self.counter(v,self.x_train,self.y_train)/float(self.y_train.size) for v in weight_history]
            val_accuracy_history = [1 - self.counter(v,self.x_val,self.y_val)/float(self.y_val.size) for v in weight_history]

            # store count history
            self.train_accuracy_histories.append(train_accuracy_history)
            self.val_accuracy_histories.append(val_accuracy_history)
 
    #### plot histories ###
    def show_histories(self,**kwargs):
        start = 0
        if 'start' in kwargs:
            start = kwargs['start']
        if self.train_portion == 1:
            self.val_cost_histories = [[] for s in range(len(self.val_cost_histories))]
        history_plotters.Setup(self.train_cost_histories,self.val_cost_histories,start)