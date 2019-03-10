# import standard plotting and animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import standard libraries
import math
import time
import copy
from inspect import signature

class Visualizer:
    '''
    animators for time series
    '''

    #### animate moving average ####
    def animate_moving_ave(self,x,y,T,savepath,**kwargs):
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
            ax1.plot(np.arange(1,x.size + 1),x,alpha = 1,c = 'k',linewidth = 2,zorder = 2);

            # plot moving average - initial conditions
            if k == 1:
                ax1.plot(np.arange(1,T + 1), y[:T], alpha = 0.75, c = 'darkorange',linewidth = 4,zorder = 3);
                
                # make vertical visual guides
                ax1.axvline(x = 1, c='deepskyblue')
                ax1.axvline(x = T, c='deepskyblue')
                
            # plot moving average - everything after and including initial conditions
            if k > 1:
                j = k-1
                # plot 
                ax1.plot(np.arange(1,T + j + 1),y[:T + j],alpha = 0.7,c = 'darkorange',linewidth = 4,zorder = 3);
                
                # make vertical visual guides
                ax1.axvline(x = j, c='deepskyblue')
                ax1.axvline(x = j + T - 1, c='deepskyblue')
                
            # label axes
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()

    #### animate range of moving average calculations ####
    def animate_moving_average_range(self,x,func,params,savepath,**kwargs):
        playback = 1
        if 'playback' in kwargs:
            playback = kwargs['playback']
            
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
                ax1.plot(np.arange(1,y.size + 1),y,alpha = 0.9,c = 'darkorange',linewidth = 3,zorder = 3);
                ax1.set_title(r'$D = $ ' + str(T))

            # label axes
            ax1.set_xlabel(r'$p$',fontsize = 13)
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        if 'fps' in kwargs:
            fps = kwargs['fps']
            
        anim.save(savepath, fps=1, extra_args=['-vcodec', 'libx264'])
        clear_output()

    #### animate range of exponential average calculations ####
    def animate_exponential_average_range(self,x,func,params,savepath,**kwargs):
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
            ax1.set_xlabel(r'$p$',fontsize = 13)
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()