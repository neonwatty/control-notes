import autograd.numpy as np
from autograd import grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
from IPython.display import clear_output
from timeit import default_timer as timer
import time

# minibatch gradient descent
def gradient_descent(g,counter,x_train,y_train,x_valid,y_valid,alpha,max_its,w,num_pts,batch_size,verbose,version):         
    if verbose:
        print('starting optimization...')
       
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    gradient = grad(g_flat)

    # set params
    num_train = y_train.size
    num_valid = y_valid.size
    
    # record histories
    weight_hist = [unflatten(w)]
    
    train_cost_hist = [g_flat(w,x_train,y_train,np.arange(num_train))]
    valid_cost_hist = [g_flat(w,x_valid,y_valid,np.arange(num_valid))]
    
    train_count_hist = [counter(unflatten(w),x_train,y_train)]
    valid_count_hist = [counter(unflatten(w),x_valid,y_valid)]

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
            grad_eval = gradient(w,x_train,y_train,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # choice of version
            if version == 'normalized':
                grad_eval = np.sign(grad_eval)
    
            # take descent step with momentum
            w = w - alpha*grad_eval
        
        # store weights
        weight_hist.append(w) 
        
        # update training cost and count
        train_cost = g_flat(w,x_train,y_train,np.arange(num_train))
        train_count = counter(unflatten(w),x_train,y_train)
        
        train_cost_hist.append(train_cost)
        train_count_hist.append(train_count)
       
        if num_valid > 0:
            valid_cost = g_flat(w,x_valid,y_valid,np.arange(num_valid))
            valid_count = counter(unflatten(w),x_valid,y_valid)        
        
            valid_cost_hist.append(valid_cost)
            valid_count_hist.append(valid_count)
            
        end = timer()

        if verbose == True:
            print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train acc = ' + str(np.round(train_count_hist[-1],4)) + ', valid acc = ' + str(np.round(valid_count_hist[-1],4)))

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
        
    return weight_hist, train_cost_hist, train_count_hist, valid_cost_hist, valid_count_hist


# minibatch RMSprop
def RMSprop(g,counter,x_train,y_train,x_valid,y_valid,alpha,max_its,w,num_pts,batch_size,verbose,version):         
    if verbose:
        print('starting optimization...')
       
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    gradient = grad(g_flat)

    # set params
    num_train = y_train.size
    num_valid = y_valid.size
    
    # record histories
    weight_hist = [unflatten(w)]
    
    train_cost_hist = [g_flat(w,x_train,y_train,np.arange(num_train))]
    valid_cost_hist = [g_flat(w,x_valid,y_valid,np.arange(num_valid))]
    
    train_count_hist = [counter(unflatten(w),x_train,y_train)]
    valid_count_hist = [counter(unflatten(w),x_valid,y_valid)]

    # how many mini-batches equal the entire dataset?
    num_batches = int(np.ceil(np.divide(num_train, batch_size)))
    avg_sq_grad = np.zeros((w.size))

    # over the line
    gamma = 0.9
    eps = 10**(-8)
    for k in range(max_its):                   
        # loop over each minibatch
        start = timer()
        train_cost = 0
        for b in range(num_batches):
            # collect indices of current mini-batch
            batch_inds = np.arange(b*batch_size, min((b+1)*batch_size, num_train))
            
            # plug in value into func and derivative
            grad_eval = gradient(w,x_train,y_train,batch_inds)
            grad_eval.shape = np.shape(w)
            
            # update exponential average of past gradients
            avg_sq_grad = gamma*avg_sq_grad + (1 - gamma)*grad_eval**2 
    
            # take descent step 
            w = w - alpha*grad_eval / (avg_sq_grad**(0.5) + eps)

        # store weights
        weight_hist.append(w) 
        
        # update training cost and count
        train_cost = g_flat(w,x_train,y_train,np.arange(num_train))
        train_count = counter(unflatten(w),x_train,y_train)
        
        train_cost_hist.append(train_cost)
        train_count_hist.append(train_count)
       
        if num_valid > 0:
            valid_cost = g_flat(w,x_valid,y_valid,np.arange(num_valid))
            valid_count = counter(unflatten(w),x_valid,y_valid)        
        
            valid_cost_hist.append(valid_cost)
            valid_count_hist.append(valid_count)
            
        end = timer()

        if verbose == True:
            print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train acc = ' + str(np.round(train_count_hist[-1],4)) + ', valid acc = ' + str(np.round(valid_count_hist[-1],4)))

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
        
    return weight_hist, train_cost_hist, train_count_hist, valid_cost_hist, valid_count_hist