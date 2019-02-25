from autograd import numpy as np

def window_series(x,order):
    # containers for input/output pairs
    x_in = []
    x_out = []
    T = x.size
    
    # window data
    for t in range(T - order):
        # get input sequence
        temp_in = x[:,t:t + order]
        x_in.append(temp_in)
        
        # get corresponding target
        temp_out = x[:,t + order]
        x_out.append(temp_out)
        
    # make array and cut out redundant dimensions
    x_in = np.array(x_in)
    x_in = x_in.swapaxes(0,1)[0,:,:].T
    x_out = np.array(x_out).T
    return x_in,x_out