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

class Animator:
    #### animate multiple runs on single regression ####
    def animate_plot(self,plotter,x,h,savepath,**kwargs):    
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

        # produce animation and save
        fps = 50
        if 'fps' in kwargs:
            fps = kwargs['fps']
        anim.save(savepath, fps=fps, extra_args=['-vcodec', 'libx264'])
        clear_output()