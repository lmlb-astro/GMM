#### Library to handle the velocity axis of spectral data cube/hyper spectral image ####
import numpy as np


def vel_to_pixel(vel, crval, dv, crpix):
    pixel = int((vel - crval)/dv + crpix + 0.5)
    return pixel


## return an array of velocities based on a velocity range and  information from the header of a hyperimage

def create_velocity_array(min_vel, max_vel, crval, dv, crpix):
    z_min = vel_to_pixel(min_vel, crval, dv, crpix)
    z_max = vel_to_pixel(max_vel, crval, dv, crpix)
    
    vel_arr = []
    for j in range(z_min,z_max):
        vel_arr.append(crval + (j-crpix)*dv)
    
    return vel_arr


## return hyper_image where the z-axis has been reduced based on a provided velocity range and the header information

def reduce_z_axis_size(data, min_vel, max_vel, crval, dv, crpix):
    z_min = vel_to_pixel(min_vel, crval, dv, crpix)
    z_max = vel_to_pixel(max_vel, crval, dv, crpix)
    data_reduced = data[z_min:z_max,:,:]
    print("The minimal pixel along the z-axis is:" + str(z_min))
    print("The maximal pixel along the z-axis is:" + str(z_max))
    
    return data_reduced















