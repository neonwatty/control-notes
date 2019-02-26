import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

class Plotter:
    def __init__(self):
        # setup figure 
        self.fig = plt.figure(figsize = (9.5,3.5))
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]);
        self.axs = [ax]
        
    def process(self,x,h,**kwargs):
        counter = len(x)
        if 'counter' in kwargs:
            counter = kwargs['counter']
            counter = min(counter,len(x))
            
        # setup figure 
        ax = self.axs[0]

        ax.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 80,linewidth = 1,zorder = 2);
        ax.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue',zorder = 2);

        # label axes
        ax.axhline(c = 'k',zorder = 0)
        ax.set_xlabel(r'$t$',fontsize = 13)
        ax.set_ylabel(r'$x_t$',fontsize=13,rotation=0)
        
                    
        # fix viewing window
        xmin = -1
        xmax = len(x) + 1
        ymin = min(x) - 2
        ymax = max(x) + 2
        
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        