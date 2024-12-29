import numpy as np
from scipy.ndimage import sobel
import torch.nn as nn
import torch
from scipy.ndimage import maximum_filter
from config import height, width
from target_image import get_target_image


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
    
    layer = state_grid[:, :, 3] # shape(height, width)

    # Apply a 3×3 sliding maximum filter
    max_in_frame = maximum_filter(layer, size = 3)

    alive = max_in_frame > 0.1 * 255
    alive = np.repeat(alive[:, :, np.newaxis], 16, axis=2)  # Shape becomes (height, width, 16)


    state_grid = state_grid * alive

    return state_grid


def compute_loss(state_grid):
    return torch.mean((state_grid - get_target_image()) ** 2)


def update(state_grid, total_loss):
    # we do a whole update step
    perception_grid = perceive(state_grid)
    updates_grid = update_grid(perception_grid)
    updated_grid = stochastic_update(state_grid, updates_grid)
    final_grid = alive_masking(updated_grid)

    tensor_grid = torch.tensor(final_grid, dtype=torch.float32)
    total_loss += compute_loss(tensor_grid)

    return final_grid, total_loss, tensor_grid