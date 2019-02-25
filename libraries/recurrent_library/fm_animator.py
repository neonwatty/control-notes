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
    animate the modulation of a sine wave signal given frequency information
    '''
    #### single dimension regression animation ####
    def animate_fm_modulation(self,t,f,fc,signal,frames,**kwargs):
        # select subset of time points to animate generation over
        inds = np.arange(0,len(t),int(len(t)/float(frames)))
        
        # produce figure
        fig = plt.figure(figsize = (9,5))
        gs = gridspec.GridSpec(3, 1) 
        ax1 = plt.subplot(gs[0]); 
        ax = plt.subplot(gs[1]); 
        ax2 = plt.subplot(gs[2]); 

        artist = fig
        
        ### set view limits
        xmin = -0.1
        xmax = np.max(t)
        
        # for frequency panel
        ymin1 = np.min(f)
        ymax1 = np.max(f) 
        ygap1 = (ymax1 - ymin1)*0.15
        ymin1 -= ygap1
        ymax1 += ygap1
        
        yminc = np.min(fc)
        ymaxc = np.max(fc) 
        ygapc = (ymaxc - yminc)*0.15
        yminc -= ygapc
        ymaxc += ygapc
        
        # for signal panel
        ymin2 = np.min(signal)
        ymax2 = np.max(signal) 
        ygap2 = (ymax2 - ymin2)*0.2
        ymin2 -= ygap2
        ymax2 += ygap2
        
        # start animation
        num_frames = len(inds)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            ax2.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # set time step
            time_step = inds[k]
            
            # plot
            ax1.plot(t[:time_step],f[:time_step],color = 'b')
            ax.plot(t[:time_step],fc[:time_step],color = 'g')

            ax2.plot(t[:time_step],signal[:time_step],color = 'fuchsia')

            # label axes and fix viewing limits
            ax1.set_title('input frequency information')
            ax1.set_xlabel('time')
            ax1.set_ylabel('frequency')
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin1,ymax1])
            
            ax.set_title('cumulative frequency')
            ax.set_xlabel('time')
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([yminc,ymaxc])
            
            ax2.set_xlabel('time')
            ax2.set_ylabel('amplitude')
            ax2.set_title('frequency modulated signal')
            ax2.set_xlim([xmin,xmax])
            ax2.set_ylim([ymin2,ymax2])
            ax2.set
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)
  
    #### single dimension regression animation ####
    def static_fm_modulation(self,t,f,signal,**kwargs):        
        # produce figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(2, 1) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]); 
        
        ### set view limits
        xmin = -0.1
        xmax = np.max(t)
        
        # for frequency panel
        ymin1 = np.min(f)
        ymax1 = np.max(f) 
        ygap1 = (ymax1 - ymin1)*0.15
        ymin1 -= ygap1
        ymax1 += ygap1
        
        # for signal panel
        ymin2 = np.min(signal)
        ymax2 = np.max(signal) 
        ygap2 = (ymax2 - ymin2)*0.2
        ymin2 -= ygap2
        ymax2 += ygap2
        
        # plot
        ax1.plot(t,f,color = 'b')
        ax2.plot(t,signal,color = 'fuchsia')

        # label axes and fix viewing limits
        ax1.set_title('input frequency information')
        ax1.set_xlabel('time')
        ax1.set_ylabel('frequency')
        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin1,ymax1])

        ax2.set_xlabel('time')
        ax2.set_ylabel('amplitude')
        ax2.set_title('frequency modulated signal')
        ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([ymin2,ymax2])
        