import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
from IPython.display import clear_output
from timeit import default_timer as timer
import time

# minibatch gradient descent
def gradient_descent(g,w,x_train,y_train,alpha,max_its,batch_size,**kwargs): 
    verbose = True
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    num_train = y_train.shape[1]
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x_train,y_train,np.arange(num_train))]
    
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_train, batch_size)))

    # over the line
    for k in range(max_its):                   
        # loop over each minibatch
        start = timer()
        train_cost = 0
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_train))
            
            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,x_train,y_train,batch_inds)
            grad_eval.shape = np.shape(w)
    
            # take descent step with momentum
            w = w - alpha*grad_eval

        end = timer()
        
        # update training and validation cost
        train_cost = g_flat(w,x_train,y_train,np.arange(num_train))

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)

        if verbose == True:
            print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train cost = ' + str(np.round(train_hist[-1][0],4)))

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
    return w_hist,train_hist

# minibatch gradient descent
def RMSprop(g,w,x_train,y_train,alpha,max_its,batch_size,**kwargs): 
    verbose = True
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
       
    # rmsprop params
    gamma=0.9
    eps=10**-8
    if 'gamma' in kwargs:
        gamma = kwargs['gamma']
    if 'eps' in kwargs:
        eps = kwargs['eps']
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # initialize average gradient
    avg_sq_grad = np.ones(np.size(w))
    if 'ave_sq_grad' in kwargs:
        avg_sq_grad = kwargs['avg_sq_grad']
    
    # record history
    num_train = y_train.shape[1]
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x_train,y_train,np.arange(num_train))]
    
    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_train, batch_size)))

    # over the line
    for k in range(max_its):                   
        # loop over each minibatch
        start = timer()
        train_cost = 0
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_train))
            
            # plug in value into func and derivative
            cost_eval,grad_eval = grad(w,x_train,y_train,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # update exponential average of past gradients
            avg_sq_grad = gamma*avg_sq_grad + (1 - gamma)*grad_eval**2 
    
            # take descent step 
            w = w - alpha*grad_eval / (avg_sq_grad**(0.5) + eps)

        end = timer()
        
        # update training and validation cost
        train_cost = g_flat(w,x_train,y_train,np.arange(num_train))

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)

        if verbose == True:
            print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train cost = ' + str(np.round(train_hist[-1][0],4)))

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
    return w_hist,train_hist,avg_sq_grad

