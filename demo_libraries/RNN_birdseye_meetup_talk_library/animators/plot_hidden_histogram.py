import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

class Plotter:
    def __init__(self):
        # setup figure
        self.fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(2,1) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 
        self.axs = [ax1,ax2]
        
    def process(self,x,h_all,**kwargs):
        counter = len(x)
        if 'counter' in kwargs:
            counter = kwargs['counter']
            counter = min(counter,len(x))
            
        ax1 = self.axs[0]
        ax2 = self.axs[1]

        # plot input sequence
        ax1.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 60,linewidth = 1,zorder = 3);
        ax1.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue');

        # strip right amount of values for histogram from dict
        h = h_all[counter]
        h_keys = list(h.keys())
        h_vals = list(h.values())

        # plot hist
        ax2.bar(h_keys,h_vals,align='center',width=0.1,edgecolor='k',color='magenta',linewidth=1.5)

        # label axes
        ax1.set_xlabel(r'$t$',fontsize = 13)
        ax1.set_ylabel(r'$x_t$',fontsize = 13,rotation = 0,labelpad = 15)
        
        # fix viewing window
        xmin = -1
        xmax = len(x) + 2
        ymin = min(x) - 2
        ymax = max(x) + 2
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])   

        # label axes
        ax2.set_xlabel(r'$h_t$',fontsize = 13)
        ax2.set_ylabel(r'count',fontsize = 13,rotation = 90,labelpad = 15)
        
        xmin = min(h_keys) - 0.2
        xmax = max(h_keys) + 0.2
        ymin = min(h_vals) - 0.1
        ymax = max(h_vals) + 0.1
        ymax = 0.25
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])