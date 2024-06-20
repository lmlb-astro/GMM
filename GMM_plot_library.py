#### Library with all the plotting functions called when running GMM models ####


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 

import astropy.io.fits as pyfits
import astropy.wcs as wcs

import GMM_functions as fGMM

import os


## inspect the data using the integrated intensity map
def inspect_intensity_map(dat, dv, wcs_val, unit_integrated_intensity, cmap_col = 'jet'):
    print(wcs_val)
    ## Produce the intensity map
    int_map = dv*np.sum(dat, axis = 0)
    
    ## plot the integrated intensity map
    fig, ax = plt.subplots()
    ax1 = fig.add_subplot(111, projection = wcs_val)
    im = ax1.imshow(int_map, origin='lower', vmin=0., cmap = cmap_col)
    
    ## set the limits of the plot
    plt.xlim([0, len(int_map[0])])
    plt.ylim([0, len(int_map)])
    
    ## add the labels to the plot
    plt.xlabel('RA [J2000]')
    plt.ylabel('DEC [J2000]',labelpad=-1.)
    
    ## Provide a colorbar for the plot
    cbar = fig.colorbar(im)
    cbar.set_label(unit_integrated_intensity, labelpad=15., rotation=270.)
    
    ax.axis('off')
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
    

## Use matplotlib to plot the input from two arrays
def plot_two_lists(list_x, list_y, label_x, label_y, plot_path = None, dpi_val = 300):
    ## create the figure
    fig, ax = plt.subplots(figsize = (10, 5))
    ax1 = fig.add_subplot(111)
    
    ## plot the two lists together
    ax1.plot(list_x,list_y,'-o')
    
    ## add the labels on the plot
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    
    ax.axis('off')
    
    ## optionally save the figure
    save_the_fig(fig, plot_path, dpi_val)
    
    plt.show()



## plot the cluster and corresponding spectra on a single figure with an option to save the figure
def plot_clusters_and_spectra(data_3d, cluster_map, vel_arr, wcs_val, normalize = False, plot_path = None, label_x = "v (km s$^{-1}$)", label_y = "T$_{mb}$ (K)", dpi_val = 300, uncertain_map = None, valrange_ax2 = None):
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
    if(valrange_ax2 is not None):
        ax2.set_ylim(valrange_ax2)
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
        
        ## define mask for the cluster
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

    
    
## Compare the average spectra and their standard deviation for two clusters
def compare_cluster_spectra(data_3d, cluster_map, vel_arr, cl_1 = 1, cl_2 = 2, normalize = True, plot_path = None, label_x = "v (km s$^{-1}$)", label_y = "T$_{mb}$ (K)", dpi_val = 300):
    ## create the figure
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    
    ## define mask for both clusters
    mask1 = np.zeros((len(cluster_map[0]), len(cluster_map)), dtype=int)
    mask1[cluster_map == cl_1] = 1
    mask2 = np.zeros((len(cluster_map[0]), len(cluster_map)), dtype=int)
    mask2[cluster_map == cl_2] = 1
        
    ## calculate the average cluster spectrum
    cluster_spectrum1, spectrum_std1 = fGMM.get_average_spectrum(data_3d, mask1)
    cluster_spectrum2, spectrum_std2 = fGMM.get_average_spectrum(data_3d, mask2)
        
        
    ## normalize the spectrum if requested
    if(normalize):
        spectrum_std1 = spectrum_std1 / np.nanmax(cluster_spectrum1)
        cluster_spectrum1 = cluster_spectrum1 / np.nanmax(cluster_spectrum1)
        
        spectrum_std2 = spectrum_std2 / np.nanmax(cluster_spectrum2)
        cluster_spectrum2 = cluster_spectrum2 / np.nanmax(cluster_spectrum2)
        
    ## define the lower and upper values of the standard deviation
    std_up1 = cluster_spectrum1 + spectrum_std1
    std_low1 = cluster_spectrum1 - spectrum_std1
    
    std_up2 = cluster_spectrum2 + spectrum_std2
    std_low2 = cluster_spectrum2 - spectrum_std2
        
    ## plot the standard deviations
    ax.fill_between(vel_arr, y1 = std_low1, y2 = std_up1, step = 'pre', alpha = 0.4)
    ax.fill_between(vel_arr, y1 = std_low2, y2 = std_up2, step = 'pre', alpha = 0.4)
            
    ## plot spectra
    ax.step(vel_arr, cluster_spectrum1, label = "Cluster " + str(cl_1 + 1), color = 'k')
    ax.step(vel_arr, cluster_spectrum2, label = "Cluster " + str(cl_2 + 1), color = 'k')
        
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    ## optionally saving the file if a path is provided
    save_the_fig(fig, plot_path, dpi_val)
    
    plt.show()

## Plot the average cluster spectrum and the standard deviation
def cluster_spectrum_std(data_3d, cluster_map, vel_arr, cl = 1, normalize = True, plot_path = None, label_x = "v (km s$^{-1}$)", label_y = "T$_{mb}$ (K)", dpi_val = 300, col = None):
    ## create the figure
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    
    ## define mask for both clusters
    mask1 = np.zeros((len(cluster_map[0]), len(cluster_map)), dtype=int)
    mask1[cluster_map == cl] = 1
        
    ## calculate the average cluster spectrum
    cluster_spectrum1, spectrum_std1 = fGMM.get_average_spectrum(data_3d, mask1)
        
        
    ## normalize the spectrum if requested
    if(normalize):
        spectrum_std1 = spectrum_std1 / np.nanmax(cluster_spectrum1)
        cluster_spectrum1 = cluster_spectrum1 / np.nanmax(cluster_spectrum1)
        
    ## define the lower and upper values of the standard deviation
    std_up1 = cluster_spectrum1 + spectrum_std1
    std_low1 = cluster_spectrum1 - spectrum_std1
        
    ## plot the standard deviations
    if(col is None):
        ax.fill_between(vel_arr, y1 = std_low1, y2 = std_up1, step = 'pre', alpha = 0.4)
    else:
        ax.fill_between(vel_arr, y1 = std_low1, y2 = std_up1, step = 'pre', alpha = 0.4, color = col)
            
    ## plot spectra
    if(col is None):
        ax.step(vel_arr, cluster_spectrum1, color = 'k')
    else:
        ax.step(vel_arr, cluster_spectrum1, color = col)
        
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    plt.title("cluster {cl}".format(cl = cl + 1))
    
    ## optionally saving the file if a path is provided
    save_the_fig(fig, plot_path, dpi_val)
    
    plt.show()