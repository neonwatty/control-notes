# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec

# import autograd functionality
import autograd.numpy as np

# various other libraries
import math
import time
import copy

def frequency_generator(kind,**kwargs):
    '''
    Function for producing different frequency inputs for frequency modulated signals.  
    Choose from 
    - random step function with multiple pieces
    - square wave based on a single sinusoid
    - random combination of cosine waves
    - random combination of 5 hidden layer maxout network
    '''
    # produce input range - preset to 5000 samples over input range [0,10]
    num_samples = 10000
    t = np.linspace(0,10,num_samples)[:,np.newaxis]
    f = np.zeros((num_samples,1))
    
    ### choose input type
    if kind == 'step':  
        f = step_generator(t,**kwargs)
    if kind == 'square':
        f = square_generator(t,**kwargs)
    if kind == 'cosine':
        f = cosine_generator(t,**kwargs)
    if kind =='maxout':
        f = maxout_generator(t,**kwargs)
        
    # return input / output frequencies
    return t,f

#### frequency modulation controllers #####
## ---- for maxout network controller ----- ##
def activation(t1,t2):
    # maxout activation
    f = np.maximum(t1,t2)
    return f

# create initial weights for arbitrary feedforward network
def initialize_maxout_weights(layer_sizes,scale):
    # container for entire weight tensor
    weights = []
    
    # loop over desired layer sizes and create appropriately sized initial 
    # weight matrix for each layer
    for k in range(len(layer_sizes)-1):
        # get layer sizes for current weight matrix
        U_k = layer_sizes[k]
        U_k_plus_1 = layer_sizes[k+1]

        # make weight matrix
        weight1 = scale*np.random.randn(U_k + 1,U_k_plus_1)
        
        # add second matrix for inner weights
        if k < len(layer_sizes)-2:
            weight2 = scale*np.random.randn(U_k + 1,U_k_plus_1)
            weights.append([weight1,weight2])
        else:
            weights.append(weight1)

    # re-express weights so that w_init[0] = omega_inner contains all 
    # internal weight matrices, and w_init = w contains weights of 
    # final linear combination in predict function
    w_init = [weights[:-1],weights[-1]]
    
    return w_init

# our normalization function
def normalize(data,data_mean,data_std):
    normalized_data = (data - data_mean)/data_std
    return normalized_data

def compute_maxout_features(x, inner_weights):
    # pad data with ones to deal with bias
    o = np.ones((np.shape(x)[0],1))
    a_padded = np.concatenate((o,x),axis = 1)
        
    # loop through weights and update each layer of the network
    for W1,W2 in inner_weights:                                  
        # output of layer activation  
        a = activation(np.dot(a_padded,W1),np.dot(a_padded,W2))  
        
        ### normalize output of activation
        # compute the mean and standard deviation of the activation output distributions
        a_means = np.mean(a,axis = 0)
        a_stds = np.std(a,axis = 0)
        
        # normalize the activation outputs
        a_normed = normalize(a,a_means,a_stds)
            
        # pad with ones for bias
        o = np.ones((np.shape(a_normed)[0],1))
        a_padded = np.concatenate((o,a_normed),axis = 1)
    
    return a_padded

# our predict function 
def predict_maxout(x,w):     
    # feature trasnsformations
    f = compute_maxout_features(x,w[0])
    
    # compute linear model
    vals = np.dot(f,w[1])
    
    # raise up to ensure nonnegative and stretch a bit
    vals = 5*vals
    valmin = np.min(vals)
    if valmin < 0:
        vals += np.abs(valmin)
    return vals

# main function for maxout network frequency generator
def maxout_generator(t,**kwargs):
    '''
    Frequency input generator using maxout network with 
    10 units per layer
    '''
    num_layers = 4
    if 'num_layers' in kwargs:
        num_layers = kwargs['num_layers']
    
    # the list defines our network architecture
    layer_sizes = [1]
    for layer in range(num_layers):
        layer_sizes += [10]
    layer_sizes += [1]

    # create initialization
    scale = 0.1
    w_init = initialize_maxout_weights(layer_sizes,scale)
    
    # create example frequency input signal
    f = predict_maxout(t,w_init)
    return f

## --- other frequency controllers --- ##
def cosine_generator(t,**kwargs):
    '''
    Makes a random combination of cosine waves as frequency input
    '''
    num_waves = 3
    if 'num_waves' in kwargs:
        num_waves = kwargs['num_waves']
        
    # loop over and generate sum of cosine waves
    f = 0
    for i in range(num_waves):
        # generate random properties of this summand in test signal
        a = np.random.rand(1)
        b = 5*np.random.rand(1)
        c = np.random.randn(1)

        # generate new addition to test signal
        s = a*np.cos(b*t + c)

        # add to full test signal
        f += s

    # normalize sinusoidal input - make it nonnegative most importantly 
    fmean = np.mean(copy.deepcopy(f))
    f -= fmean
    fmax = np.max(f)
    fmin = np.min(f)
    f += np.abs(fmin)
    f = np.array([v/(fmax - fmin) for v in f]) 
    f *= np.maximum(1,9*np.random.rand(1)) 
    f+=0.5
    return f

def square_generator(t,**kwargs):
    '''
    Makes a random frequency square wave frequency input 
    '''
    square_freq = 0.2
    if 'square_freq' in kwargs:
        square_freq = kwargs['square_freq']
    
    # produce square wave based on cosine
    f = np.sign(np.maximum(0,np.cos(np.pi*square_freq*t))) + 1
    return f
    
def step_generator(t,**kwargs):
    '''
    Makes a random piecewise constant step frequency input.  Note 
    that the steps always take on integer values.
    '''
    # lets make random piecewise frequency
    num_steps = 2
    if 'num_steps' in kwargs:
        num_steps = kwargs['num_steps'] - 1
    
    # pick num_pieces random locations for step ledges
    r = np.random.permutation(len(t))[:num_steps]
    r = np.sort(r)

    # generate random level per step
    levels = np.round(5*np.random.rand(num_steps + 1)) + 1
    f = np.ones((t.size,1))

    # set each chunk to appropriate level
    f[:r[0]] *= levels[0]  # set first chunk to appropriate level
    for n in range(1,len(r)-1):
        f[r[n-1]:r[n]] *= levels[n]
    f[r[-1]:] *= levels[-1]
    
    return f

#### generate test signal and plot both input frequency and correct / incorrect output ####
def fm_tester(kind,**kwargs):
    ### generate input frequency signal
    t,f = frequency_generator(kind,**kwargs)
    
    ### from input frequency construct correct and incorrect fm output
    # correctly - using hidden state variable
    phase_in = 2*np.pi*np.cumsum(f)*float(t[1] - t[0])
    correct_signal = np.sin(phase_in)

    # incorrectly - using no hidden state sequence
    phase_in_2 = 2*np.pi*f
    incorrect_signal= np.sin(phase_in_2)
    return t,f,correct_signal, incorrect_signal

    '''
    #### create figure and plot 
    fig = plt.figure(figsize = (8,4))
    gs = gridspec.GridSpec(3, 1) 

    # setup current axis
    ax1 = plt.subplot(gs[0]);
    ax2 = plt.subplot(gs[1]);
    ax3 = plt.subplot(gs[2]);

    # plot stuff
    ax1.plot(t,f,color = 'k')
    ax1.set_title('input frequency signal')
    ax2.plot(t,y,c = 'b')
    ax2.set_title('correctly modulated output signal')
    ax3.plot(t,y2,c = 'r')
    ax3.set_title('incorrectly modulated output signal')
    plt.show()
    '''
