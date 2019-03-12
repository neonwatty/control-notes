import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

class Plotter:
    def plot_input_with_hidden_together(self,x,h,**kwargs):
        title = ''
        hidden_name = ''
        ylabel = ''
        counter = len(h)
        if 'title' in kwargs:
            title = kwargs['title']
        if 'hidden_name' in kwargs:
            hidden_name = kwargs['hidden_name']
        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
        if 'counter' in kwargs:
            counter = kwargs['counter']
            counter = min(counter,len(h))
            
        # fix viewing window
        xmin = -1
        xmax = len(x) + 1
        ymin = min(min(x),min(h)) - 2
        ymax = max(max(x),max(h)) + 2
        
        # setup figure 
        fig = plt.figure(figsize = (9.5,3.5))
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]);

        ax.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 80,linewidth = 1,zorder = 2);
        ax.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue',zorder = 2);

        ax.scatter(np.arange(1,counter+1),h[:counter],c = 'fuchsia',edgecolor = 'w',s = 80,linewidth = 1,zorder = 3);
        ax.plot(np.arange(1,counter+1),h[:counter],alpha = 0.5,c = 'fuchsia',zorder = 3);

        # label axes
        ax.axhline(c = 'k',zorder = 0)
        ax.set_xlabel(r'$t$',fontsize = 13)
        ax.set_ylabel(ylabel,fontsize=13)
        ax.legend(['$x_t$ - input','$h_t$ - ' + hidden_name],loc='upper left')
        ax.set_title(title,fontsize = 14)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        self.axs = [ax]
        self.fig = fig

    def plot_input_with_hidden_separate(self,x,h,**kwargs):
        counter = len(h)
        if 'counter' in kwargs:
            counter = kwargs['counter']
            counter = min(counter,len(h))
            
        # plot the result
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(2,1) 
        ax2 = plt.subplot(gs[0]); 
        ax1 = plt.subplot(gs[1]); 

        # plot input sequence
        ax1.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 80,linewidth = 1,zorder = 2);
        ax1.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue',zorder = 2);

        ax2.scatter(np.arange(1,counter+1),h[:counter],c = 'fuchsia',edgecolor = 'w',s = 80,linewidth = 1,zorder = 3);
        ax2.plot(np.arange(1,counter+1),h[:counter],alpha = 0.5,c = 'fuchsia',zorder = 3);

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
        
        self.axs = [ax1,ax2]
        self.fig = fig
        
    def plot_input(self,x):    
        # setup figure 
        fig = plt.figure(figsize = (9.5,3.5))
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]);

        ax.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 80,linewidth = 1,zorder = 2);
        ax.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue',zorder = 2);

        # label axes
        ax.axhline(c = 'k',zorder = 0)
        ax.set_xlabel(r'$t$',fontsize = 13)
        ax.set_ylabel(r'$x_t$',fontsize=13,rotation=0)
        
        self.axs = [ax]
        self.fig = fig

    def plot_hidden_histogram(self,x,h_all,**kwargs):
        counter = len(x)
        if 'counter' in kwargs:
            counter = kwargs['counter']
            counter = min(counter,len(x))
            
        # plot the result
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(2,1) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 

        # plot input sequence
        ax1.scatter(np.arange(1,counter+1),x[:counter],c = 'mediumblue',edgecolor = 'w',s = 60,linewidth = 1,zorder = 3);
        ax1.plot(np.arange(1,counter+1),x[:counter],alpha = 0.5,c = 'mediumblue');

        # strip right amount of values for histogram from dict
        h = h_all[counter]
        h_keys = list(h.keys())
        h_vals = list(h.values())
#         h_plot = []
#         for k in h_keys:
#             h_plot.append(h[k])

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
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin,ymax])
        
        self.axs = [ax1,ax2]
        self.fig = fig