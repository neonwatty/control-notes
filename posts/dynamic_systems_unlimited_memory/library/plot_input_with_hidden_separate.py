import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

class Plotter:
    def __init__(self):
        # setup figure 
        self.fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(2,1) 
        ax2 = plt.subplot(gs[0]); 
        ax1 = plt.subplot(gs[1]); 
        self.axs = [ax1,ax2]
        
    def process(self,x,h,**kwargs):
        counter = len(h)
        if 'counter' in kwargs:
            counter = kwargs['counter']
            counter = min(counter,len(h))
            
        ax1 = self.axs[0]
        ax2 = self.axs[1]
        
        # plot input sequence
        ax1.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 80,linewidth = 1,zorder = 2);
        ax1.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue',zorder = 2);

        ax2.scatter(np.arange(1,counter+1),h[:counter],c = 'darkorange',edgecolor = 'w',s = 80,linewidth = 1,zorder = 3);
        ax2.plot(np.arange(1,counter+1),h[:counter],alpha = 0.5,c = 'darkorange',zorder = 3);

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
        ax2.set_xlabel(r'$t$',fontsize = 13)
        ax2.set_ylabel(r'$h_t$',fontsize = 13,rotation = 0,labelpad = 15)
        
        ymin = min(h) - 2
        ymax = max(h) + 2
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])  
        