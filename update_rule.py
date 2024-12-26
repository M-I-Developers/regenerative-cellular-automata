import numpy as np
from scipy.ndimage import sobel
import torch.nn as nn
import torch
from config import height, width
from scipy.ndimage import maximum_filter


def perceive(state_grid):
    # apply sobel filters
    grad_x = sobel(state_grid, 0)
    grad_y = sobel(state_grid, 1)

    # concatenate the state's information + info about the neighbours
    perception_grid = np.concatenate((state_grid, grad_x, grad_y), axis=2)

    return perception_grid


def update_rule(perception_vector):

    perception_vector = torch.tensor(perception_vector, dtype=torch.float32)
    # The following code operates on a single cell's perception vector
    # (This is why we return to 16 dimension)
    first_layer = nn.Linear(48, 128)

    # apply activation function
    first_layer_result = nn.functional.relu(first_layer(perception_vector))

    second_layer = nn.Linear(128, 16)
    weights = torch.tensor(np.zeros((16, 128)), dtype=torch.float32)


    # apply weights
    with torch.no_grad():
        second_layer.weight.copy_(weights)

        second_layer_result = second_layer(first_layer_result)


        return second_layer_result.numpy()


def update_grid(perception_grid):
    update_grid = np.empty((height, width, 16))

    for h in range(height):
        for w in range(width):
            update_grid[h, w, :] = update_rule(perception_grid[h][w])

    return update_grid


def stochastic_update(state_grid, update_grid):
    # we want to update only a part of the cells,
    # so we either update the whole cell or none of it(this is why
    #  we do not specify the third dimension)
    rand_mask = np.random.rand(height, width) < 0.5

    # repetas the value on the nely added dimension
    rand_mask = np.repeat(rand_mask[:, :, np.newaxis], 16, axis=2)  # Shape becomes (height, width, 16)

    update_grid = update_grid * rand_mask

    return state_grid + update_grid


def alive_masking(state_grid):
    # we want to check if there is at least one alive neighbour cell
    # we're doing this by taking the max alpha channel from the neighbours
    #
    # alive = np.max(state_grid[:, :, 3], (3, 3)) > 0.1
    # rand_mask = np.repeat(alive[:, :, np.newaxis], 16, axis=2)
    layer = state_grid[:, :, 3] # shape(height, width)

    # Apply a 3×3 sliding maximum filter
    max_in_frame = maximum_filter(layer, size = 3)

    alive = max_in_frame > 0.1
    alive = np.repeat(alive[:, :, np.newaxis], 16, axis=2)  # Shape becomes (height, width, 16)


    state_grid = state_grid * alive

    return state_grid