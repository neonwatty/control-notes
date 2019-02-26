import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

class Plotter:
    def __init__(self,**kwargs):
        # setup figure 
        self.fig = plt.figure(figsize = (9.5,3.5))
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]);
        self.axs = [ax]
        
        self.title = ''
        self.hidden_name = ''
        self.ylabel = ''
        
        if 'title' in kwargs:
            self.title = kwargs['title']
        if 'hidden_name' in kwargs:
            self.hidden_name = kwargs['hidden_name']
        if 'ylabel' in kwargs:
            self.ylabel = kwargs['ylabel']
        
    def process(self,x,h,**kwargs):
        counter = len(h)
        if 'counter' in kwargs:
            counter = kwargs['counter']
            counter = min(counter,len(h))
            
        # fix viewing window
        xmin = -1
        xmax = len(x) + 1
        ymin = min(min(x),min(h)) - 2
        ymax = max(max(x),max(h)) + 2
        
        ax = self.axs[0]
        
        ax.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 40,linewidth = 1,zorder = 3);
        ax.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue',zorder = 3);

        ax.scatter(np.arange(1,counter+1),h[:counter],c = 'fuchsia',edgecolor = 'w',s = 120,linewidth = 1,zorder = 2);
        ax.plot(np.arange(1,counter+1),h[:counter],alpha = 0.5,c = 'fuchsia',zorder = 2);

        # label axes
        ax.axhline(c = 'k',zorder = 0)
        ax.set_xlabel(r'$t$',fontsize = 13)
        ax.set_ylabel(self.ylabel,fontsize=13)
        ax.legend(['$x_t$ - input','$h_t$ - ' + self.hidden_name],loc='upper left')
        ax.set_title(self.title,fontsize = 14)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])