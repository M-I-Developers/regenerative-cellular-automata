import numpy as np
from scipy.ndimage import sobel

def perceive(state_grid):

    # apply sobel filters
    grad_x = sobel(state_grid, 0) 
    grad_y = sobel(state_grid, 1)

    # concatenate the state's information + info about the neighbours
    perception_grid = np.concatenate((state_grid, grad_x, grad_y), axis = 2)
    
    return perception_grid