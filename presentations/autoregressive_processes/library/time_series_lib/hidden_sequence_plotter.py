# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib import gridspec

# import autograd functionality
import numpy as np
import math
import time
import copy

class Visualizer:
    '''
    Plotting tools for looking at driver / hidden sequence pairs
    '''             

    ######## functions for regression visualization ########
    # plot regression data
    def show_sequences(self,seq1,seq2,**kwargs):
        # create figure and plot data
        fig = plt.figure(figsize = (9,3.5))
        gs = gridspec.GridSpec(1, 1) 

        # setup current axis
        ax = plt.subplot(gs[0]);
            
        # plot sequence 1 
        ax.plot(np.arange(len(seq1)),seq1,c = [1,0.8,0.5],zorder = 1,linewidth=3.5)

        # plot sequence 2
        if np.size(seq2) > 0:
            ax.plot(np.arange(len(seq2)),seq2,c = [0.5,0.7,1],zorder = 2,linewidth=2.5)
            ax.set_ylim([min(-1,min(seq2))-1,max(1,max(seq2))+1])

        # label axes
        ax.set_xlabel('step')
        ax.set_ylabel('value')
        
        # adjust viewing range
        ax.set_xlim([-1,len(seq1)])

        # draw horizontal and vertical lines
        ax.axhline(linewidth=0.5, color='k',zorder = 0)

        plt.show()
        
    def plot_pair(self,seq1,seq2):    
        # normalize each sequence to output [-1,1]
        mean1 = np.mean(copy.deepcopy(seq1))
        std1 = np.std(seq1)
        seq1 = [(v - mean1)/std1 for v in seq1]

        mean2 = np.mean(copy.deepcopy(seq2))
        std2 = np.std(seq2)
        seq2 = [(v - mean2)/std2 for v in seq2]

        # initialize figure
        fig = plt.figure(figsize = (9,5))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(2,1) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);

        # now plot each, one per panel
        ax1.scatter(np.arange(len(seq1)),seq1,edgecolor = 'k')  
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('on/off switch')
        ax1.set_title('on/off switch')

        ax2.plot(seq2)
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('heat measurement')
        ax2.set_title('continuous heat measurement')

        plt.show()