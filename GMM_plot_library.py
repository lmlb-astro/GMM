#### Library with all the plotting functions called when running GMM models ####


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 

import astropy.io.fits as pyfits
import astropy.wcs as wcs

import os


## inspect the data using the integrated intensity map
def inspect_intensity_map(dat, dv, wVal, unit_integrated_intensity):
    ## Produce the intensity map
    int_map = dv*np.sum(dat, axis=0)
    
    ## plot the integrated intensity map
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(111, projection=wVal)
    im = ax1.imshow(int_map, origin='lower', vmin=0., cmap = 'jet')
    
    ## set the limits and labels of the plot
    plt.xlim([0, len(int_map[0])])
    plt.ylim([0, len(int_map)])
    
    plt.xlabel('RA [J2000]')
    plt.ylabel('DEC [J2000]',labelpad=-1.)
    
    ## Provide a colorbar for the plot
    cbar = fig.colorbar(im)
    cbar.set_label(unit_integrated_intensity, labelpad=15.,rotation=270.)
    
    ax.axis('off')
    plt.show()


## Use matplotlib to plot the input from two arrays
def plot_two_lists(list_x,list_y,label_x,label_y):
    plt.plot(list_x,list_y,'-o')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.show()


## plot the cluster and corresponding spectra on a single figure with an option to save the figure
def plot_clusters_and_spectra(data_3d, cluster_map, vel_arr, label_x, label_y, wVal, save_opt, plot_path, save_name, dpi_val):
    ## define the two axes used for the plot
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(121, projection=wVal)
    ax2 = fig.add_subplot(122)
    
    ## determine the range of values associated with the clusters
    min_val = int(np.nanmin(cluster_map)+0.5)
    max_val = int(np.nanmax(cluster_map)+0.5)
    
    ## loop over clusters to plot spectra
    mask = np.zeros((len(cluster_map[0]),len(cluster_map)),dtype=int)
    for j in range(min_val,max_val+1):
        print("The routine is currently plotting cluster {index}".format(index = j))
        
        ## define mask for the cluster and extend for third dimension
        mask = np.zeros((len(cluster_map[0]), len(cluster_map)), dtype=int)
        mask[cluster_map==j] = 1
        mask_3d = np.broadcast_to(mask, data_3d.shape)
        
        ## calculate the average cluster spectrum
        cluster = data_3d.copy()
        cluster[mask_3d==0] = np.nan
        cluster_spectrum = np.nanmean(cluster,axis=(1,2))
        
        ## plot spectrum
        p = ax2.step(vel_arr, cluster_spectrum, label = "Cluster " + str(j+1))
        
        ## store color of plotted spectrum, make custom color and set zeros in mask to nan
        color_val = p[0].get_color()
        color_map = colors.ListedColormap([color_val])
        mask = mask.astype(float)
        mask[mask==0] = np.nan
        
        ## plot the map
        im = ax1.imshow(mask, origin='lower', cmap = color_map)
    
    ## finalize axis 1
    ax1.set_xlim([0, len(cluster_map[0])])
    ax1.set_ylim([0, len(cluster_map)])
    
    ax1.set_xlabel('RA [J2000]')
    ax1.set_ylabel('DEC [J2000]',labelpad=-1.)
    
    ## finalize axis 2
    ax2.set_xlabel(label_x)
    ax2.set_ylabel(label_y)
    ax2.legend()
    
    ax.axis('off')
    plt.tight_layout()
    
    if(save_opt):
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)
        plt.savefig(plot_path+save_name,dpi=dpi_val)
    
    plt.show





