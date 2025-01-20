import torch
import torch.nn.functional as F
from scipy.ndimage import sobel
from nn import model, trained_model
from config import height, width
from nn import compute_loss

def perceive(state_grid):
    state_grid_array = state_grid.clone().numpy()

    # apply sobel filters
    grad_x = sobel(state_grid_array, 0)
    grad_y = sobel(state_grid_array, 1)

    grad_x = torch.from_numpy(grad_x).float()
    grad_y = torch.from_numpy(grad_y).float()

    # concatenate the state's information + info about the neighbours
    perception_grid = torch.cat((state_grid, grad_x, grad_y), dim=2)

    return perception_grid


def update_rule(perception_vector, train):
    # go through the nn
    if train:
        result = model(perception_vector)
    else:
        result = trained_model(perception_vector)
    return result


def update_grid(perception_grid, train):
    update_grid = torch.empty((height, width, 16))
    
    for h in range(height):
        for w in range(width):
            update_grid[h, w, :] = update_rule(perception_grid[h, w, :], train)

    return update_grid


def stochastic_update(state_grid, update_grid):
    # we want to update only a part of the cells,
    # so we either update the whole cell or none of it
    rand_mask = (torch.rand(height, width) < 0.5).unsqueeze(-1) # (height, width, 1)
    
    # broadcast the mask along the channels dimension
    rand_mask = rand_mask.expand(-1, -1, 16)  # (height, width, channels)

    update_grid = update_grid * rand_mask

    return state_grid + update_grid


def alive_masking(state_grid):
    # we want to check if there is at least one alive neighbour cell
    # we're doing this by taking the max alpha channel from the neighbours
    layer = state_grid[:, :, 3].clone().detach()

    # Apply a 3×3 sliding maximum filter
    max_in_frame = F.max_pool2d(layer.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    alive = (max_in_frame.squeeze(0).squeeze(0) > 0.1).unsqueeze(-1)  # (height, width, 16)

    return alive


def update(state_grid, train=True):
    # we do a whole update step
    with torch.no_grad():
        pre_life_mask = alive_masking(state_grid)
        perception_grid = perceive(state_grid)

    updates_grid = update_grid(perception_grid, train)
    loss = compute_loss(state_grid, updates_grid)

    with torch.no_grad():
        updated_grid = stochastic_update(state_grid, updates_grid)
        
        post_life_mask = alive_masking(updated_grid)
        life_mask = pre_life_mask & post_life_mask
        final_grid = updated_grid * life_mask

    return final_grid, loss