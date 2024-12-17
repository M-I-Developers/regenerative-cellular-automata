import numpy as np
from scipy.ndimage import convolve

def perceive(state_grid):
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1 ]]
    
    sobel_y = sobel_x.transpose()


    # apply sobel filters
    grad_x = convolve(state_grid, sobel_x, mode = 'constant')
    grad_y = convolve(state_grid, sobel_y, mode = 'constant')

    # concatenate the state's information + info about the neighbours
    perception_grid = np.concatenate((state_grid, grad_x, grad_y), axis = 2)
    
    
    return perception_grid