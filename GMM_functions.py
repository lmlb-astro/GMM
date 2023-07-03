#### Library with functions to execute the GMM modeling ####


import numpy as np
from sklearn.mixture import GaussianMixture

import astropy.io.fits as pyfits
import astropy.wcs as wcs

import velocity_axis_datacube as vax


## prepare the data for input into the GMM model

def prepare_data(data, min_vel, max_vel, crval, dv, crpix):
    
    ## resample data, so that the GMM can handle it
    ## i.e. resample the data to a single array of the spectra
    len_resample = len(data[0])*len(data[0][0])
    data_resample = data.reshape(len(data),len_resample)
    print(data_resample.shape)
    
    
    ## create an array to keep track of the indices
    index_arr = [int(j) for j in range(0,len_resample)]
    index_arr = np.array(index_arr)
    print("Dimensions of index_arr: " + str(index_arr.shape))
    print(index_arr)
    
    
    ## normalize the data: maximum value = 1
    max_vals = np.nanmax(data_resample,axis=0)
    data_resample = data_resample / max_vals[np.newaxis, :]
    ## verification
    print("Dimensions of max_vals: " + str(max_vals.shape))
    print(max_vals)
    print("The maximum value is: " + str(np.nanmax(data_resample)))
    print("The minimum value is: " + str(np.nanmin(data_resample)))
    
    
    ## remove nan-data from data and index array
    index_arr = index_arr[~np.isnan(max_vals)]
    mask = np.full((len_resample),True)
    mask[np.isnan(max_vals)] = False
    data_resample = data_resample[:,mask]
    
    ## transpose the arrays to provide as input for the GMM
    index_arr = index_arr.transpose()
    data_resample = data_resample.transpose()
    
    return index_arr, data_resample
    print("Final dimensions of index_arr: " + str(index_arr.shape))
    print("Final dimensions of data_resample: " + str(data_resample.shape))
    print(data_resample)


### run the GMM for a set number of components

def run_GMM(sample_data, num_comps, seed, threshold, max_iter):
    model = GaussianMixture(n_components=num_comps, init_params='kmeans', covariance_type='full', random_state=seed, tol=threshold, max_iter=max_iter).fit(sample_data)
    return model


## Maps the different clusters on the observed map in different colors
def map_spatial_cluster_distribution(cluster_inds_arr,index_arr,len_x_orig_map,len_y_orig_map):
    cluster_map = np.zeros((len_x_orig_map,len_y_orig_map),dtype=float)
    cluster_map[cluster_map==0] = np.nan
    for cluster_ind, ind in zip(cluster_inds_arr,index_arr):
        pos_x = ind%len_x_orig_map
        pos_y = int(ind/len_x_orig_map)
        cluster_map[pos_y][pos_x] = cluster_ind
    return cluster_map








