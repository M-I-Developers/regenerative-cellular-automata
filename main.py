import numpy as np
import sklearn.neural_network as nn
from update_rule import perceive, update_rule, stochastic_update, alive_masking, update_grid
from config import height, width

state_grid = np.zeros(shape=(height, width, 16))

# plant center seed
seed_height = height // 2
seed_width = width // 2
state_grid[seed_height, seed_width, 3:] = 1

perception_grid = perceive(state_grid)
updates_grid = update_grid(perception_grid)
updated_grid = stochastic_update(state_grid, updates_grid)
final_grid = alive_masking(updated_grid)


print(final_grid)