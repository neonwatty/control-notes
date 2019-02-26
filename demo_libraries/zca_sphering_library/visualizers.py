import numpy as np

# import standard plotting and animation
from matplotlib import gridspec
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########## plotting functionality ############
def show_images(X,**kwargs):
    '''
    Function for plotting input images, stacked in columns of input X.
    '''
    cmap = 'gray'
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
        
    # plotting mechanism taken from excellent answer from stack overflow: https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
    plt.figure(figsize = (9,3))
    gs1 = gridspec.GridSpec(5, 14)
    gs1.update(wspace=0, hspace=0.05) # set the spacing between axes. 

    # shape of square version of image
    square_shape = int((X.shape[0])**(0.5))

    for i in range(min(70,X.shape[1])):
        # plot image in panel
        ax = plt.subplot(gs1[i])
        im = ax.imshow(255 - np.reshape(X[:,i],(square_shape,square_shape)),cmap = cmap)

        # clean up panel
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()