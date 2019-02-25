# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
import autograd.numpy as np

# import standard libraries
import math
import time
import copy
from inspect import signature

class Visualizer:
    '''
    animators for time series
    '''
        
    #### animate running average ####
    def animate_running_ave(self,x,y,**kwargs):
        # produce figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,7,1]) 
        ax = plt.subplot(gs[0]); ax.axis('off')
        ax1 = plt.subplot(gs[1]); 
        ax2 = plt.subplot(gs[2]); ax2.axis('off')
        artist = fig
        
        # view limits
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
            
        # start animation
        num_frames = len(y) + 1
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot x
            ax1.plot(np.arange(1,x.size + 1),x,alpha = 1,c = 'k',linewidth = 1,zorder = 2);

            # plot running average - everything after and including initial conditions
            if k >= 1:
                j = k-1
                # plot 
                a = np.arange(1,j+2)
                b = y[:j+1]
                #print (a.shape,b.shape)
                ax1.plot(np.arange(1,j+2),y[:j+1],alpha = 0.7,c = 'fuchsia',linewidth = 3,zorder = 3);
                ax1.scatter(1,y[0],alpha = 0.7,c = 'fuchsia',linewidth = 3,zorder = 3);

            # label axes
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)
    
    #### animate moving average ####
    def animate_moving_ave(self,x,y,T,**kwargs):
        # produce figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,7,1]) 
        ax = plt.subplot(gs[0]); ax.axis('off')
        ax1 = plt.subplot(gs[1]); 
        ax2 = plt.subplot(gs[2]); ax2.axis('off')
        artist = fig
        
        # view limits
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
            
        # start animation
        num_frames = len(y) - T + 1
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot x
            ax1.plot(np.arange(1,x.size + 1),x,alpha = 1,c = 'k',linewidth = 1,zorder = 2);

            # plot moving average - initial conditions
            if k == 1:
                ax1.plot(np.arange(1,T + 1),y[:T],alpha = 0.75,c = 'fuchsia',linewidth = 3,zorder = 3);
                
                # make vertical visual guides
                ax1.axvline(x = 1)
                ax1.axvline(x = T)
                
            # plot moving average - everything after and including initial conditions
            if k > 1:
                j = k-1
                # plot 
                ax1.plot(np.arange(1,T + j + 1),y[:T + j],alpha = 0.7,c = 'fuchsia',linewidth = 3,zorder = 3);
                
                # make vertical visual guides
                ax1.axvline(x = j)
                ax1.axvline(x = j + T - 1)
                
            # label axes
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)
 
    #### animate range of moving average calculations ####
    def animate_moving_average_range(self,x,func,params,**kwargs):
        # produce figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,7,1]) 
        ax = plt.subplot(gs[0]); ax.axis('off')
        ax1 = plt.subplot(gs[1]); 
        ax2 = plt.subplot(gs[2]); ax2.axis('off')
        artist = fig
        
        # view limits
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
            
        # start animation
        num_frames = len(params)+1
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot x
            ax1.plot(np.arange(1,x.size + 1),x,alpha = 1,c = 'k',linewidth = 1,zorder = 2);

            # create y
            if k == 0:
                T = params[0]
                y = func(x,T)
                #ax1.plot(np.arange(1,y.size + 1),y,alpha = 1,c = 'fuchsia',linewidth = 3,zorder = 3);
                ax1.set_title(r'Original data')

            if k > 0:
                T = params[k-1]
                y = func(x,T)
                ax1.plot(np.arange(1,y.size + 1),y,alpha = 0.9,c = 'fuchsia',linewidth = 3,zorder = 3);
                ax1.set_title(r'$\mathcal{O} = $ ' + str(T))

            # label axes
            ax1.set_xlabel(r'$t$',fontsize = 13)
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)

    #### animate range of exponential average calculations ####
    def animate_exponential_average_range(self,x,func,params,**kwargs):
        # produce figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,7,1]) 
        ax = plt.subplot(gs[0]); ax.axis('off')
        ax1 = plt.subplot(gs[1]); 
        ax2 = plt.subplot(gs[2]); ax2.axis('off')
        artist = fig
        
        # view limits
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
            
        # start animation
        num_frames = len(params)+1
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot x
            ax1.plot(np.arange(1,x.size + 1),x,alpha = 1,c = 'k',linewidth = 1,zorder = 2);

            # create y
            if k == 0:
                alpha = params[0]
                y = func(x,alpha)
                #ax1.plot(np.arange(1,y.size + 1),y,alpha = 1,c = 'fuchsia',linewidth = 3,zorder = 3);
                ax1.set_title(r'Original data')

            if k > 0:
                alpha = params[k-1]
                y = func(x,alpha)
                ax1.plot(np.arange(1,y.size + 1),y,alpha = 0.9,c = 'fuchsia',linewidth = 3,zorder = 3);
                ax1.set_title(r'$\alpha = $ ' + str(np.round(alpha,2)))

            # label axes
            ax1.set_xlabel(r'$t$',fontsize = 13)
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)