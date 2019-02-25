# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only

# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
from IPython.display import clear_output
import time
import copy


# simple first order taylor series visualizer
class Visualizer:
    '''
    Visualize how an input sequence is generated - from the perspective of regression.  
    ''' 
     
    # window sequences to produce input/output data
    def window_my_sequence(self):
        # containers for input/output pairs
        x = []
        y = []
        input_window_size = 1

        # window data
        count = 0
        for t in range(len(self.input_sequence) - input_window_size):
            # get input sequence
            temp_in = self.input_sequence[t:t + input_window_size]
            x.append(temp_in)

            # get corresponding target
            temp_target = self.input_sequence[t + input_window_size]
            y.append(temp_target)
            count+=1

        # reshape each 
        x = np.asarray(x)
        x.shape = (np.shape(x)[0:2])
        y = np.asarray(y)
        y.shape = (len(y),)

        return x,y

    # animate the method
    def seq_as_regression(self,input_sequence,**kwargs):
        # run in input sequence
        self.input_sequence = input_sequence
        
        # transform input sequence to simple regression dataset
        x,y = self.window_my_sequence()

        # initialize figure
        fig = plt.figure(figsize = (14,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
        ax1 = plt.subplot(gs[0]); 
        ax2 = plt.subplot(gs[1]);       
        artist = fig
        
        ### setup viewing area for left and right plot
        # setup left plot viewing area - this is where the original sequence is plotted
        xmin_left = -0.5
        xmax_left = len(self.input_sequence) + 0.5
        ymin_left = min(self.input_sequence)
        ymax_left = max(self.input_sequence)
        ygap_left = (ymax_left - ymin_left)*0.25
        ymin_left -= ygap_left
        ymax_left += ygap_left
        
        # for right plot - of regression data
        xmin_right = min(copy.deepcopy(x))
        xmax_right = max(copy.deepcopy(x))
        xgap_right = abs(xmax_right - xmin_right)*0.5
        xmin_right -= xgap_right
        xmax_right += xgap_right
        
        
        ymin_right = min(copy.deepcopy(y))
        ymax_right = max(copy.deepcopy(y))
        ygap_right = abs(ymax_right - ymin_right)*0.5    
        ymin_right -= ygap_right
        ymax_right += ygap_right  
        
        # animation sub-function
        P = len(self.input_sequence)+1
        print ('beginning animation rendering...')
        def animate(k):
            ax1.cla()
            ax2.cla()
                        
            # print rendering update
            if k == P-1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            #### plot data ####
            # plot sequence in left panel, as regression data in right
            ax1.plot(np.arange(0,k),self.input_sequence[:k],c = 'k',zorder = 0,linewidth=2)
            ax1.scatter(np.arange(0,k),self.input_sequence[:k],c = 'k',zorder = 2,s=100,edgecolor = 'w',linewidth=2)
            
            # scatter plot as regression data in right panel
            if k > 0:
                cs = 0*np.ones((k,3))
                weights = [1/float(v) for v in range(1,k)]
                weights = weights + [0]
                weights = np.asarray(weights)
                #weights = np.flipud(weights)
                weights.shape = (len(weights),1)
                cs2 = cs*weights
                ax2.scatter(x[:k-1],y[:k-1],c = cs,zorder = 2,s=120,edgecolor = 'w',linewidth=2)

            # fix viewing limits
            ax1.set_xlim([xmin_left,xmax_left])
            ax1.set_ylim([ymin_left,ymax_left])
                
            # fix viewing limits
            ax2.set_xlim([xmin_right,xmax_right])
            ax2.set_ylim([ymin_right,ymax_right])

            # set titles
            ax1.set_title('sequence view (as data generated)',fontsize = 14)
            ax2.set_title('regression view (as data generated)',fontsize = 14)
   
            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=P, interval=P, blit=True)

        return(anim)

    
    # animate the method
    def single_plot(self,x,y,**kwargs):      
        # initialize figure
        fig = plt.figure(figsize = (10,4))
        artist = fig
        
        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1,1) 
        ax1 = plt.subplot(gs[0]);# ax1.set_aspect('equal')
        artist = fig
        
        ### setup viewing area for left and right plot
        # for right plot - of regression data
        xmin_right = min(copy.deepcopy(x))
        xmax_right = max(copy.deepcopy(x))
        xgap_right = abs(xmax_right - xmin_right)*0.25
        xmin_right -= xgap_right
        xmax_right += xgap_right
        
        ymin_right = min(copy.deepcopy(y))
        ymax_right = max(copy.deepcopy(y))
        ygap_right = abs(ymax_right - ymin_right)*0.4
        ymin_right -= ygap_right
        ymax_right += ygap_right  
        
        # animation sub-function
        P = len(x)
        print ('beginning animation rendering...')
        def animate(k):
            ax1.cla()
                        
            # print rendering update
            if k == P-1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            #### plot data ####
            # plot sequence in left panel, as regression data in right
            ax1.scatter(x[:k],y[:k],c = 'k',zorder = 2,s=120,edgecolor = 'w',linewidth=2)
                
            # fix viewing limits
            ax1.set_xlim([xmin_right,xmax_right])
            ax1.set_ylim([ymin_right,ymax_right])

            # set titles
            ax1.set_title('watch the points go by...',fontsize = 14)
   
            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=P, interval=P, blit=True)

        return(anim)
