import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
from IPython.display import clear_output
from timeit import default_timer as timer
import time

# gradient descent
def gradient_descent(g,w,x_train,y_train,x_val,y_val,alpha,max_its,**kwargs): 
    verbose = True
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    num_val = y_val.size
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x_train,y_train)]
    val_hist = []
    if num_val > 0:
        val_hist.append(g_flat(w,x_val,y_val))
        
    # over the line
    alpha_choice = 0
    for k in range(1,max_its+1): 
        # check if diminishing steplength rule used
        if alpha == 'diminishing':
            alpha_choice = 1/float(k)
        else:
            alpha_choice = alpha
            
        # take a single descent step
        start = timer()

        # plug in value into func and derivative
        cost_eval,grad_eval = grad(w,x_train,y_train)
        grad_eval.shape = np.shape(w)
    
        # take descent step with momentum
        w = w - alpha_choice*grad_eval
        
        end = timer()
        
        # update training and validation cost
        train_cost = g_flat(w,x_train,y_train)
        val_cost = np.nan

        # record weight update, train and val costs
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)
        if num_val > 0:
            val_cost = g_flat(w,x_val,y_val)
            val_hist.append(val_cost)

        if verbose == True:
            print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train cost = ' + str(np.round(train_hist[-1][0],4)) )#+ ', val cost = ' + str(np.round(val_hist[-1],4)))

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
        #time.sleep(1.5)
        #clear_output()
    return w_hist,train_hist,val_hist


# zero order coordinate search
def coordinate_descent_zorder(g,w,x_train,y_train,x_val,y_val,alpha,max_its,**kwargs):  
    verbose = True
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)

    # record history
    num_val = y_val.size
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,x_train,y_train)]
    val_hist = []
    if num_val > 0:
        val_hist.append(g_flat(w,x_val,y_val))
        
    # run coordinate search
    N = np.size(w)
    alpha_choice = 0
    for k in range(1,max_its+1):        
        # check if diminishing steplength rule used
        if alpha == 'diminishing':
            alpha_choice = 1/float(k)
        else:
            alpha_choice = alpha
        
        # random shuffle of coordinates
        c = np.random.permutation(N)
        
        # forming the direction matrix out of the loop
        train_cost = train_hist[-1]
        
        # loop over each coordinate direction
        for n in range(N):
            start = timer()

            direction = np.zeros((N,1)).flatten()
            direction[c[n]] = 1
    
            # evaluate all candidates
            evals =  [g_flat(w + alpha_choice*direction,x_train,y_train)]
            evals.append(g_flat(w - alpha_choice*direction,x_train,y_train))
            evals = np.array(evals)

            # if we find a real descent direction take the step in its direction
            ind = np.argmin(evals)
            if evals[ind] < train_cost:
                # take step
                w = w + ((-1)**(ind))*alpha_choice*direction
                train_cost = evals[ind]
                
            # record weight update, train and val costs
            w_hist.append(unflatten(w))
            train_hist.append(train_cost)
            val_cost = np.nan
            if num_val > 0:
                val_cost = g_flat(w,x_val,y_val)
                val_hist.append(val_cost)

            end = timer()

            if verbose == True:
                print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train cost = ' + str(np.round(train_hist[-1][0],4)) )#+ ', val cost = ' + str(np.round(val_hist[-1][0],4)))

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
        #time.sleep(1.5)
        #clear_output()
        
    return w_hist,train_hist,val_hist
