import matplotlib.pyplot as plt
from matplotlib import gridspec
from autograd import numpy as np 
import copy

# plot regression data
def plot_train_gen_sequences(runner,x_series,**kwargs): 
    ### get best weights ###
    x = runner.x
    order = x.shape[0]
   
    # get model and trained weights
    model = runner.model
    train_hist = runner.train_cost_histories[0]
    val_hist = runner.val_cost_histories[0]
    w = 0
    if np.size(val_hist) > 0:
        ind = np.argmin(val_hist)
        w = runner.weight_histories[0][ind]
    else:
        ind = np.argmin(train_hist)
        w = runner.weight_histories[0][ind]
  
    ### extract training sequence, make generated sequence ###
    # generate predictions for training
    x_train = x[:,runner.train_inds]
    pred_train = model(x_train,w)

    # pad 'order' number of nans to front of predictions for plotting
    pad = np.array([np.nan for n in range(order)])[np.newaxis,:]
    pred_train = np.hstack((pad,pred_train))

    # make generated data
    x_gen = list([v for v in pred_train[:,-1 - order + 1:]][0])
    for i in range(np.size(runner.val_inds)):
        # get previous prediction
        xp = np.array(x_gen[-1 - order + 1:])
        if np.ndim(xp) == 1:
            xp = xp[:,np.newaxis]
        
        # pass through model to generate next prediction
        pred = model(xp,w).flatten()
        x_gen.append(list(pred))

    # array-afy generated entries
    x_gen = np.array(x_gen[order:]).T
    
    # pad 'order' number of nans to front of predictions for plotting
    pad = np.array([np.nan for n in range(np.size(pred_train))])[np.newaxis,:]
    pred_gen = np.hstack((pad,x_gen))

    ### plot original, training, and validation sequences ###
    # plotting colors
    colors = [[0,0.7,1],'magenta']
    
    # create figure and plot data
    fig = plt.figure(figsize = (9,3.5))
    gs = gridspec.GridSpec(1, 1) 

    # setup current axis
    ax = plt.subplot(gs[0]);

    # plot original sequence 
    ax.plot(np.arange(np.size(x_series)),x_series.flatten(),c = 'k',zorder = 1,linewidth=3)
    
    # plot training fit
    ax.plot(np.arange(np.size(pred_train)),pred_train.flatten(),c = colors[0],zorder = 1,linewidth=3)
    
    # plot validation fit
    ax.plot(np.arange(np.size(pred_gen)),pred_gen.flatten(),c = colors[1],zorder = 1,linewidth=3)
    
    
    # cleanup panel
    ymax = np.max(copy.deepcopy(x_series))
    ymin = np.min(copy.deepcopy(x_series))
    ygap = (ymax - ymin)*0.2
    ymax += ygap
    ymin -= ygap
    #ax.set_xlim([-1,np.size(x_series)])
    #ax.set_ylim([ymin,ymax])
    ax.set_xlabel('step')
    ax.set_ylabel('value')
    
    # draw horizontal and vertical lines
    ax.axhline(linewidth=0.5, color='k',zorder = 0)
    plt.show()
    
# plot regression data
def plot_train_val_sequences(runner,x_series,**kwargs): 
    ### get best weights ###
    x = runner.x
    order = x.shape[0]
    
    # get model and trained weights
    model = runner.model
    train_hist = runner.train_cost_histories[0]
    val_hist = runner.val_cost_histories[0]
    w = 0
    if np.size(val_hist) > 0:
        ind = np.argmin(val_hist)
        w = runner.weight_histories[0][ind]
    else:
        ind = np.argmin(train_hist)
        w = runner.weight_histories[0][ind]
        
    ### extract training and validation sequences ###
    # generate predictions for training
    x_train = x[:,runner.train_inds]
    pred_train = model(x_train,w)

    # pad 'order' number of nans to front of predictions for plotting
    pad = np.array([np.nan for n in range(order)])[np.newaxis,:]
    pred_train = np.hstack((pad,pred_train))

    # generate predictions for validation
    x_val = x[:,runner.val_inds]
    pred_val = model(x_val,w)

    # pad 'order' number of nans to front of predictions for plotting
    pad = np.array([np.nan for n in range(np.size(pred_train))])[np.newaxis,:]
    pred_val = np.hstack((pad,pred_val))

    ### plot original, training, and validation sequences ###
    # plotting colors
    colors = [[0,0.7,1],[1,0.8,0.5]]
    
    # create figure and plot data
    fig = plt.figure(figsize = (9,3.5))
    gs = gridspec.GridSpec(1, 1) 

    # setup current axis
    ax = plt.subplot(gs[0]);

    # plot original sequence 
    ax.plot(np.arange(np.size(x_series)),x_series.flatten(),c = 'k',zorder = 1,linewidth=3)
    
    # plot training fit
    ax.plot(np.arange(np.size(pred_train)),pred_train.flatten(),c = colors[0],zorder = 1,linewidth=3)
    
    # plot validation fit
    ax.plot(np.arange(np.size(pred_val)),pred_val.flatten(),c = colors[1],zorder = 1,linewidth=3)
    
    
    # cleanup panel
    ymax = np.max(copy.deepcopy(x_series))
    ymin = np.min(copy.deepcopy(x_series))
    ygap = (ymax - ymin)*0.2
    ymax += ygap
    ymin -= ygap
    #ax.set_xlim([-1,np.size(x_series)])
    #ax.set_ylim([ymin,ymax])
    ax.set_xlabel('step')
    ax.set_ylabel('value')
    
    # draw horizontal and vertical lines
    ax.axhline(linewidth=0.5, color='k',zorder = 0)
    plt.show()
    
# plot regression data
def plot_sequences(seq1,**kwargs):
    seq2 = []
    if 'seq2' in kwargs:
        seq2 = kwargs['seq2']
        
    # create figure and plot data
    fig = plt.figure(figsize = (9,3.5))
    gs = gridspec.GridSpec(1, 1) 

    # setup current axis
    ax = plt.subplot(gs[0]);

    # plot sequence 1 
    ax.plot(np.arange(np.size(seq1)),seq1.flatten(),c = 'k',zorder = 1,linewidth=3)

    # plot sequence 2 if it was given
    if np.size(seq2) > 0:
        ax.plot(np.arange(np.size(seq2)),seq2.flatten(),c = 'fuchsia',zorder = 2,linewidth=2.5)
      
    # cleanup panel
    ymax = np.max(copy.deepcopy(seq1))
    ymin = np.min(copy.deepcopy(seq1))
    ygap = (ymax - ymin)*0.2
    ymax += ygap
    ymin -= ygap
    ax.set_xlim([-1,np.size(seq1)])
    ax.set_ylim([ymin,ymax])
    ax.set_xlabel('step')
    ax.set_ylabel('value')
    
    # draw horizontal and vertical lines
    ax.axhline(linewidth=0.5, color='k',zorder = 0)
    plt.show()