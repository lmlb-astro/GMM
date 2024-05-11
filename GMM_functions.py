#### Library with functions to execute the GMM modeling ####


import numpy as np
import time

import astropy.io.fits as pyfits
import astropy.wcs as wcs

from sklearn.mixture import GaussianMixture

import velocity_axis_datacube as vax

#### parameters related to fit_GMM ####

seed_val = 312
threshold = 0.001
gmm_iter = 1000

#####################################


## prepare the data for input into the GMM model
def prepare_data(data, min_vel, max_vel, crval, dv, crpix):
    
    ## resample data, so that the GMM can handle it
    ## i.e. resample the data to a single array of the spectra
    len_resample = data.shape[1] * data.shape[2] #len(data[0])*len(data[0][0])
    data_resample = data.reshape(len(data),len_resample)
    print("Dimensions of resampled data: {shape}".format(shape = data_resample.shape))
    
    ## create an array to keep track of the indices
    index_arr = np.arange(len_resample)
    
    ## normalize the data: maximum value = 1
    max_vals = np.nanmax(data_resample, axis=0)
    data_resample = data_resample / max_vals[np.newaxis, :]
    
    ## remove nan-data from data and index array
    index_arr = index_arr[~np.isnan(max_vals)]
    mask = np.full((len_resample),True)
    mask[np.isnan(max_vals)] = False
    data_resample = data_resample[:,mask]
    
    ## transpose the arrays to provide as input for the GMM
    index_arr = index_arr.transpose()
    data_resample = data_resample.transpose()
    
    return index_arr, data_resample



## Maps the different clusters on the observed map in different colors
def map_spatial_cluster_distribution(cluster_inds_arr, index_arr, len_x_orig_map, len_y_orig_map):
    ## initialize the map for the cluster tags
    cluster_map = np.zeros((len_x_orig_map, len_y_orig_map), dtype=float)
    cluster_map[cluster_map==0] = np.nan
    
    ## allocate cluster index values to pixels in the map based on the index array
    for cluster_ind, ind in zip(cluster_inds_arr,index_arr):
        pos_x = ind%len_x_orig_map
        pos_y = int(ind/len_x_orig_map)
        cluster_map[pos_y][pos_x] = cluster_ind
    
    return cluster_map



## loop over number of clusters for the GMM modeling
def fit_GMM(data_in, n_comps_min, n_comps_max):
    ## initialize the storage lists
    n_comps_list = []
    time_list = []
    bic_list = []
    bic_min = None
    best_model = None
    
    ## loop over the number of components to be fitted
    for i in range(n_comps_min, n_comps_max):
        ## perform the GMM fitting
        print("Calculating GMM for {index} number of components".format(index = i))
        start_time = time.time()
        temp_model = GaussianMixture(n_components = i, 
                                     init_params = 'kmeans', 
                                     covariance_type = 'full', 
                                     random_state = seed_val, 
                                     tol = threshold, 
                                     max_iter = gmm_iter).fit(data_in)
        end_time = time.time()
        time_list.append(end_time - start_time)
        
        ## calculate the Bayesian Information Criterion (BIC)
        temp_bic = temp_model.bic(data_in)
        
        ## store the best fitting model
        if(i == n_comps_min or temp_bic < bic_min):
            best_model = temp_model
            bic_min = temp_bic
        n_comps_list.append(i)
        bic_list.append(temp_bic)
        
    return n_comps_list, bic_list, time_list, best_model







