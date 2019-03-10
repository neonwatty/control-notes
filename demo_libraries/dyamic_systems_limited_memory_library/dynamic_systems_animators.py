# import standard plotting and animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
import matplotlib.ticker as ticker

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
    def animate_system(self,x,y,T,savepath,**kwargs):
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
            ax1.scatter(np.arange(1,x.size + 1),x,c = 'k',edgecolor = 'w',s = 40,linewidth = 1,zorder = 3);
            ax1.plot(np.arange(1,x.size + 1),x,alpha = 0.5,c = 'k',zorder = 3);
                            
            # plot moving average - initial conditions
            if k == 1:
                # plot x
                ax1.scatter(np.arange(1,T + 1), y[:T],c = 'darkorange',edgecolor = 'w',s = 120,linewidth = 1,zorder = 2);
                ax1.plot(np.arange(1,T + 1), y[:T],alpha = 0.5,c = 'darkorange',zorder = 2);
                            
                # make vertical visual guides
                ax1.axvline(x = 1, c='deepskyblue')
                ax1.axvline(x = T, c='deepskyblue')
                
            # plot moving average - everything after and including initial conditions
            if k > 1:
                j = k-1
                # plot 
                ax1.scatter(np.arange(1,T + j + 1),y[:T + j],c = 'darkorange',edgecolor = 'w',s = 120,linewidth = 1,zorder = 2);
                ax1.plot(np.arange(1,T + j + 1),y[:T + j],alpha = 0.5,c = 'darkorange',zorder = 2);
               
                
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
    def animate_system_range(self,x,func,params,savepath,**kwargs):
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
            ax1.scatter(np.arange(1,x.size + 1),x,c = 'k',edgecolor = 'w',s = 40,linewidth = 1,zorder = 3);
            ax1.plot(np.arange(1,x.size + 1),x,alpha = 0.5,c = 'k',zorder = 3);
        
            # create y
            if k == 0:
                T = params[0]
                y = func(x,T)
                ax1.set_title(r'Original data')

            if k > 0:
                T = params[k-1]
                y = func(x,T)
                
                ax1.scatter(np.arange(1,y.size + 1),y,c = 'darkorange',edgecolor = 'w',s = 120,linewidth = 1,zorder = 2);
                ax1.plot(np.arange(1,y.size + 1),y,alpha = 0.5,c = 'darkorange',zorder = 2);
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
       
    #### animate vector system with heatmap ####
    def animate_vector_system(self,x,D,model,func,savepath,**kwargs):
        x = np.array(x)
        h,old_bins = func([0])
        
        bins = []
        for i in range(len(old_bins)-1):
            b1 = old_bins[i]
            b2 = old_bins[i+1]
            n = (b1 + b2)/2
            n = np.round(n,2)
            bins.append(n)
        
        y = model(x,D,func)
        num_windows = len(y) - 1
        
        # produce figure
        fig = plt.figure(figsize = (11,10))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1,7,1],height_ratios=[0.75,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        ax4 = plt.subplot(gs[3]); ax4.axis('off')
        ax5 = plt.subplot(gs[4]); 
        ax6 = plt.subplot(gs[5]); ax6.axis('off')
        
        artist = fig
        
        # view limits
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
        
        # make colormap
#         a,b = np.meshgrid(np.arange(num_windows+1),np.arange(len(bins)-1))
#         s = ax1.pcolormesh(a, b, np.array(y).T,cmap = 'hot',vmin = 0,vmax = 1) #,edgecolor = 'k') # hot, gist_heat, cubehelix
#         ax1.cla(); ax1.axis('off');
#         fig.colorbar(s, ax=ax5)

        
        # start animation
        num_frames = len(x) - D + 2
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax2.cla()
            ax5.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot x
            ax2.scatter(np.arange(1,x.size + 1),x,c = 'k',edgecolor = 'w',s = 80,linewidth = 1,zorder = 3);
            ax2.plot(np.arange(1,x.size + 1),x,alpha = 0.5,c = 'k',zorder = 3);
                            
            # plot moving average - initial conditions
            if k == 0:
                # plot x
                ax2.scatter(np.arange(1,D + 1), x[:D],c = 'darkorange',edgecolor = 'w',s = 200,linewidth = 1,zorder = 2);
                ax2.plot(np.arange(1,D + 1), x[:D],alpha = 0.5,c = 'darkorange',zorder = 2);
                            
                # make vertical visual guides
                ax2.axvline(x = 1, c='deepskyblue')
                ax2.axvline(x = D, c='deepskyblue')
                
                # plot histogram
                self.plot_heatmap(ax5,y[:2],bins,num_windows)
                
            # plot moving average - everything after and including initial conditions
            if k > 0:
                j = k
                # plot 
                ax2.scatter(np.arange(j,D + j),x[j-1:D + j - 1],c = 'darkorange',edgecolor = 'w',s = 200,linewidth = 1,zorder = 2);
                ax2.plot(np.arange(j,D + j),x[j-1:D + j - 1],alpha = 0.5,c = 'darkorange',zorder = 2);
               
                
                # make vertical visual guides
                ax2.axvline(x = j, c='deepskyblue')
                ax2.axvline(x = j + D - 1, c='deepskyblue')
                
                # plot histogram
                self.plot_heatmap(ax5,y[:j+1],bins,num_windows)
                
            # label axes
            ax2.set_xlim([xmin,xmax])
            ax2.set_ylim([ymin,ymax])
            
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()       
        

    def plot_heatmap(self,ax,y,bins,num_windows):
        y=np.array(y).T
        
        ### plot ###
        num_chars,num_samples = y.shape
        num_chars += 1
        a,b = np.meshgrid(np.arange(num_samples),np.arange(num_chars))

        ### y-axis Customize minor tick labels ###
        # make custom labels
        num_bins = len(bins)+1
        y_ticker_range = np.arange(0.5,num_bins,10).tolist()
        new_bins = [bins[v] for v in range(0,len(bins),10)]
        y_char_range = [str(s) for s in new_bins]

        # assign major or minor ticklabels? - chosen major by default
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticker_range))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(y_char_range))
        ax.xaxis.set_ticks_position('bottom') # the rest is the same
        
        ax.set_xticks([],[])
        ax.set_yticks([],[])

        ax.set_ylabel('values',rotation = 90,fontsize=15)
        ax.set_xlabel('window',fontsize=15)

    #     ax.set_title(title,fontsize = 15)
        cmap = 'hot_r'
        #cmap = 'RdPu'
        s = ax.pcolormesh(a, b, 4*y,cmap = cmap,vmin = 0,vmax = 1) #,edgecolor = 'k') # hot, gist_heat, cubehelix

        ax.set_ylim([-1,len(bins)])
        ax.set_xlim([0,num_windows])

       # for i in range(len(bins)):
       #     ax.hlines(y=i, xmin=0, xmax=num_windows, linewidth=1, color='k',alpha = 0.75)
         
    
    #### animate vector system with heatmap ####
    def animate_vector_histogram(self,x,D,model,func,savepath,**kwargs):
        x = np.array(x)
        h,old_bins = func([0])
        
        bins = []
        for i in range(len(old_bins)-1):
            b1 = old_bins[i]
            b2 = old_bins[i+1]
            n = (b1 + b2)/2
            n = np.round(n,2)
            bins.append(n)
        
        y = model(x,D,func)
        num_windows = len(y) - 1
        
        # produce figure
        fig = plt.figure(figsize = (11,10))
        gs = gridspec.GridSpec(3, 3, width_ratios=[1,7,1],height_ratios=[1,1,1.5]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        axa = plt.subplot(gs[3]); axa.axis('off')
        axb = plt.subplot(gs[7]); 
        axc = plt.subplot(gs[5]); axc.axis('off')
      
        ax4 = plt.subplot(gs[6]); ax4.axis('off')
        ax5 = plt.subplot(gs[4]); 
        ax6 = plt.subplot(gs[8]); ax6.axis('off')
        
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
        num_frames = len(x) - D + 2
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax2.cla()
            ax5.cla()
            axb.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot x
            ax2.scatter(np.arange(1,x.size + 1),x,c = 'k',edgecolor = 'w',s = 80,linewidth = 1,zorder = 3);
            ax2.plot(np.arange(1,x.size + 1),x,alpha = 0.5,c = 'k',zorder = 3);
                            
            # plot moving average - initial conditions
            if k == 0:
                # plot x
                ax2.scatter(np.arange(1,D + 1), x[:D],c = 'darkorange',edgecolor = 'w',s = 200,linewidth = 1,zorder = 2);
                ax2.plot(np.arange(1,D + 1), x[:D],alpha = 0.5,c = 'darkorange',zorder = 2);
                            
                # make vertical visual guides
                ax2.axvline(x = 1, c='deepskyblue')
                ax2.axvline(x = D, c='deepskyblue')
                
                # plot histogram
                self.plot_histogram(ax5,y[0],bins)
                self.plot_heatmap(axb,y[:2],bins,num_windows)

            # plot moving average - everything after and including initial conditions
            if k > 0:
                j = k
                # plot 
                ax2.scatter(np.arange(j,D + j),x[j-1:D + j - 1],c = 'darkorange',edgecolor = 'w',s = 200,linewidth = 1,zorder = 2);
                ax2.plot(np.arange(j,D + j),x[j-1:D + j - 1],alpha = 0.5,c = 'darkorange',zorder = 2);
               
                
                # make vertical visual guides
                ax2.axvline(x = j, c='deepskyblue')
                ax2.axvline(x = j + D - 1, c='deepskyblue')
                
                # plot histogram
                self.plot_histogram(ax5,y[j],bins)
                
                # plot histogram
                self.plot_heatmap(axb,y[:j+1],bins,num_windows)
                
            # label axes
            ax2.set_xlim([xmin,xmax])
            ax2.set_ylim([ymin,ymax])
            ax2.set_xlabel(r'$p$',fontsize=14)
            ax2.set_ylabel(r'$x_p$',rotation=0,fontsize=14)
           
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()    
        
    def plot_histogram(self,ax,h,bins,**kwargs):
        # plot hist
        ax.bar(bins,h,align='center',width=0.1,edgecolor='k',color='magenta',linewidth=1.5)

        # label axes
        ax.set_xlabel(r'$values$',fontsize = 13)
        ax.set_ylabel(r'count',fontsize = 13,rotation = 90,labelpad = 15)
        
        ymin = 0
        xmin = min(bins) - 0.1
        xmax = max(bins) + 0.1
        ymax = 0.5
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
       
    
    #### animate spectrogram construction ####
    def animate_dct_spectrogram(self,x,D,model,func,savepath,**kwargs):
        # produce heatmap
        y = model(x,D,func)
        num_windows = y.shape[1]-1
        
        # produce figure
        fig = plt.figure(figsize = (12,8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1,7,1],height_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        ax4 = plt.subplot(gs[3]); ax4.axis('off')
        ax5 = plt.subplot(gs[4]); 
        ax5.set_yticks([],[])
        ax5.axis('off') 
                
        ax6 = plt.subplot(gs[5]); ax6.axis('off')
        artist = fig
        
        # view limits for top panel
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
        
        vmin = np.min(np.log(1 + y).flatten())
        vmax = np.max(np.log(1 + y).flatten())

        
        # start animation
        num_frames = len(x) - D + 2
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax2.cla()
            ax5.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot signal
            ax2.plot(np.arange(1,x.size + 1),x,alpha = 0.5,c = 'k',zorder = 3);
                            
            # plot moving average - initial conditions
            if k == 0:
                # plot x
                ax2.plot(np.arange(1,D + 1), x[:D],alpha = 0.5,c = 'magenta',zorder = 2,linewidth=8);
                            
                # plot spectrogram
                ax5.imshow(np.log(1 + y[:,:1]),aspect='auto',cmap='jet',origin='lower',vmin = vmin, vmax = vmax)
    
            # plot moving average - everything after and including initial conditions
            if k > 0:
                j = k
                # plot 
                ax2.plot(np.arange(j,D + j),x[j-1:D + j - 1],alpha = 0.5,c = 'magenta',zorder = 2,linewidth=8);
                
                # plot histogram
                ax5.imshow(np.log(1 + y[:,:j+1]),aspect='auto',cmap='jet',origin='lower', vmin = vmin, vmax = vmax)
                
            # label axes
            ax2.set_xlim([xmin,xmax])
            ax2.set_ylim([ymin,ymax])
            ax2.set_xlabel(r'$p$',fontsize=14)
            ax2.set_ylabel(r'$x_p$',rotation=0,fontsize=14)
            
            ax5.set_xlim([0,num_windows])
           
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()    
 
    #### animate spectrogram construction ####
    def animate_mlp_outputs(self,x,D,model,func,savepath,**kwargs):
        # produce heatmap
        y = model(x,D,func)
        num_windows = y.shape[1]-1
        
        # produce figure
        fig = plt.figure(figsize = (12,8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1,7,1],height_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax2 = plt.subplot(gs[1]); 
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        ax4 = plt.subplot(gs[3]); ax4.axis('off')
        ax5 = plt.subplot(gs[4]); 
        ax5.set_yticks([],[])
        ax5.axis('off') 
                
        ax6 = plt.subplot(gs[5]); ax6.axis('off')
        artist = fig
        
        # view limits for top panel
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
        
        vmin = np.min(np.log(1 + y).flatten())
        vmax = np.max(np.log(1 + y).flatten())

        
        # start animation
        num_frames = len(x) - D + 2
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax2.cla()
            ax5.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot signal
            ax2.plot(np.arange(1,x.size + 1),x,alpha = 0.5,c = 'k',zorder = 3);
                            
            # plot moving average - initial conditions
            if k == 0:
                # plot x
                ax2.plot(np.arange(1,D + 1), x[:D],alpha = 0.5,c = 'magenta',zorder = 2,linewidth=8);
                            
                # plot spectrogram
                ax5.imshow(np.log(1 + y[:,:1]),aspect='auto',cmap='jet',origin='lower',vmin = vmin, vmax = vmax)
    
            # plot moving average - everything after and including initial conditions
            if k > 0:
                j = k
                # plot 
                ax2.plot(np.arange(j,D + j),x[j-1:D + j - 1],alpha = 0.5,c = 'magenta',zorder = 2,linewidth=8);
                
                # plot histogram
                ax5.imshow(np.log(1 + y[:,:j+1]),aspect='auto',cmap='jet',origin='lower', vmin = vmin, vmax = vmax)
                
            # label axes
            ax2.set_xlim([xmin,xmax])
            ax2.set_ylim([ymin,ymax])
            ax2.set_xlabel(r'$p$',fontsize=14)
            ax2.set_ylabel(r'$x_p$',rotation=0,fontsize=14)
            
            ax5.set_xlim([0,num_windows])
           
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()    
        
        