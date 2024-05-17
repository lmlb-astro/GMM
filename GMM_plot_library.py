#### Library with all the plotting functions called when running GMM models ####


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 

import astropy.io.fits as pyfits
import astropy.wcs as wcs

import GMM_functions as fGMM

import os


## inspect the data using the integrated intensity map
def inspect_intensity_map(dat, dv, wcs_val, unit_integrated_intensity):
    ## Produce the intensity map
    int_map = dv*np.sum(dat, axis = 0)
    
    ## plot the integrated intensity map
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(111, projection = wcs_val)
    im = ax1.imshow(int_map, origin='lower', vmin=0., cmap = 'jet')
    
    ## set the limits of the plot
    plt.xlim([0, len(int_map[0])])
    plt.ylim([0, len(int_map)])
    
    ## add the labels to the plot
    plt.xlabel('RA [J2000]')
    plt.ylabel('DEC [J2000]',labelpad=-1.)
    
    ## Provide a colorbar for the plot
    cbar = fig.colorbar(im)
    cbar.set_label(unit_integrated_intensity, labelpad=15.,rotation=270.)
    
    ax.axis('off')
    plt.show()


## Use matplotlib to plot the input from two arrays
def plot_two_lists(list_x, list_y, label_x, label_y):
    ## plot the two lists together
    plt.plot(list_x,list_y,'-o')
    
    ## add the labels on the plot
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    
    plt.show()


## function to save the figure
def save_the_fig(fig, plot_path, dpi_val = 300):
    ## verify there is a path to save the figure
    if(plot_path):
        ## If the directory does not exist, create it
        directory = "/".join(plot_path.split("/")[:-1])
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        ## save the figure
        fig.savefig(plot_path, dpi = dpi_val)


## plot the cluster and corresponding spectra on a single figure with an option to save the figure
def plot_clusters_and_spectra(data_3d, cluster_map, vel_arr, wcs_val, normalize = False, plot_path = None, label_x = "v (km s$^{-1}$)", label_y = "T$_{mb}$ (K)", dpi_val = 300, uncertain_map = None):
    ## define the two axes used for the plot
    fig, ax = plt.subplots(figsize = (10, 5))
    ax1 = fig.add_subplot(121, projection = wcs_val)
    ax2 = fig.add_subplot(122)
    
    ## determine the range of values associated with the clusters
    min_val, max_val = fGMM.get_cluster_range(cluster_map)
    
    ## loop over clusters to plot spectra
    for j in range(min_val,max_val+1):
        print("The routine is currently plotting cluster {index}".format(index = j))
        
        ## define mask for the cluster and extend for third dimension
        mask = np.zeros((len(cluster_map[0]), len(cluster_map)), dtype=int)
        mask[cluster_map == j] = 1
        
        ## calculate the average cluster spectrum
        cluster_spectrum, spectrum_std = fGMM.get_average_spectrum(data_3d, mask)
        
        
        ## normalize the spectrum if requested
        if(normalize):
            cluster_spectrum = cluster_spectrum / np.nanmax(cluster_spectrum)
        
        ## plot spectrum
        p = ax2.step(vel_arr, cluster_spectrum, label = "Cluster " + str(j+1))
        
        ## store color of plotted spectrum, make custom color and set zeros in mask to nan
        color_val = p[0].get_color()
        color_map = colors.ListedColormap([color_val])
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        
        ## plot the map
        im = ax1.imshow(mask, origin = 'lower', cmap = color_map)
    
    ## Indicate the pixels with low certainty in black
    if(uncertain_map is not None):
        im = ax1.imshow(uncertain_map, origin = 'lower', cmap = 'binary', vmin = 0., vmax = 1.)
    
    ## finalize axis 1
    ax1.set_xlim([0, len(cluster_map[0])])
    ax1.set_ylim([0, len(cluster_map)])
    
    ax1.set_xlabel('RA [J2000]')
    ax1.set_ylabel('DEC [J2000]',labelpad=-1.)
    
    ## finalize axis 2
    ax2.set_xlabel(label_x)
    ax2.set_ylabel(label_y)
    ax2.legend()
    
    ## finalizing the axes
    ax.axis('off')
    plt.tight_layout()
    
    
    ## optionally saving the file if a path is provided
    save_the_fig(fig, plot_path, dpi_val)
    
    plt.show()


## Plot the average spectra and their standard deviation for each cluster
def plot_cluster_spectra(data_3d, cluster_map, vel_arr, normalize = False, plot_path = None, label_x = "v (km s$^{-1}$)", label_y = "T$_{mb}$ (K)", dpi_val = 300):
    ## determine the range of values associated with the clusters
    min_val, max_val = fGMM.get_cluster_range(cluster_map)
    diff = max_val + 1 - min_val
    
    ## determine the number of subplots
    fig, axs = plt.subplots(nrows = diff//3 + 1, ncols = 3, figsize = (3*(diff//3), 6))
    
    ## loop over all clusters
    for j in range(min_val,max_val+1):
        print("The routine is currently plotting cluster {index}".format(index = j))
        
        ## define mask for the cluster and extend for third dimension
        mask = np.zeros((len(cluster_map[0]), len(cluster_map)), dtype=int)
        mask[cluster_map == j] = 1
        
        ## calculate the average cluster spectrum
        cluster_spectrum, spectrum_std = fGMM.get_average_spectrum(data_3d, mask)
        
        
        ## normalize the spectrum if requested
        if(normalize):
            spectrum_std = spectrum_std / np.nanmax(cluster_spectrum)
            cluster_spectrum = cluster_spectrum / np.nanmax(cluster_spectrum)
        
        ## define the lower and upper values of the standard deviation
        std_up = cluster_spectrum + spectrum_std
        std_low = cluster_spectrum - spectrum_std
        
        ## plot the standard deviation
        axs[(j-min_val)//3, (j-min_val)%3].fill_between(vel_arr, y1 = std_low, y2 = std_up, step = 'pre')
            
        ## plot spectrum
        axs[(j-min_val)//3, (j-min_val)%3].step(vel_arr, cluster_spectrum, label = "Cluster " + str(j+1), color = 'k')
        
        if((j-min_val)//3 == diff//3):
            axs[(j-min_val)//3, (j-min_val)%3].set_xlabel(label_x)
        if((j-min_val)%3 == 0):
            axs[(j-min_val)//3, (j-min_val)%3].set_ylabel(label_y)
        
    plt.tight_layout()
    plt.show()


