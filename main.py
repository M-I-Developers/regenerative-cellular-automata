import numpy as np
import sklearn.neural_network as nn
from update_rule import perceive

height = 256
width = 256
state_grid = np.zeros(shape=(height, width, 16))

# plant center seed
seed_height = height // 2
seed_width = width // 2
state_grid[seed_height, seed_width, 3:] = 1 

perception_grid = perceive(state_grid)
print(perception_grid.shape)