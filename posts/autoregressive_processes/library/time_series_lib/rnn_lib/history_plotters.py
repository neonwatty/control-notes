# import standard plotting and animation
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Setup:
    def __init__(self,train_cost_histories,val_cost_histories,start):
        # plotting colors
        self.colors = [[0,0.7,1],[1,0.8,0.5]]
        self.plot_cost_histories(train_cost_histories,val_cost_histories,start)
 
    #### compare cost function histories ####
    def plot_cost_histories(self,train_cost_histories,val_cost_histories,start):        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(train_cost_histories)):
            train_history = train_cost_histories[c]
            val_history = val_cost_histories[c]

            # plot train cost function history
            ax.plot(np.arange(start,len(train_history),1),train_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[0],label = 'train cost') 
            
            if np.size(val_history) > 0:
                # plot test cost function history
                ax.plot(np.arange(start,len(val_history),1),val_history[start:],linewidth = 3*(0.8)**(c),color = self.colors[1],label = 'test cost') 

        # clean up panel / axes labels
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'train vs validation cost histories'
        ax.set_title(title,fontsize = 18)
        
        # plot legend
        anchor = (1,1)
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
        ax.set_xlim([start - 0.5,len(train_history) - 0.5]) 
        plt.show()