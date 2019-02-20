# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from library.JSAnimation_slider_only import IPython_display_slider_only
from IPython.display import clear_output
import math
import time
from matplotlib import gridspec
import copy
import numpy as np

class Animator:
    #### animate multiple runs on single regression ####
    def animate_plot(self,plotter,x,h,**kwargs):    
        num_frames = len(x)
        if 'num_frames' in kwargs:
            num_frames = min(num_frames,kwargs['num_frames'])

        ## construct figure
        artist = plotter.fig

        # start animation
        inds = np.arange(0,len(x),int(len(x)/float(num_frames)))
        num_frames = len(inds)        

        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            for ax in plotter.axs:
                ax.cla()

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # plot
            c = inds[k]
            plotter.process(x,h,counter=c)
            return artist,

        anim = animation.FuncAnimation(artist,animate,frames=num_frames, interval=num_frames, blit=True)

        return(anim)