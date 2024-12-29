import numpy as np
import random
from config import height, width, image
from update_rule import update
from plot import animate_automata
from target_image import load_image

img = load_image(image, height, width)

# state_grid = np.zeros(shape=(height, width, 16))
state_grid = img

# plant center seed
# seed_height = height // 2
# seed_width = width // 2
# state_grid[seed_height, seed_width, 3:] = 1

# sample random number of steps
steps = random.randint(5, 10)

animate_automata(state_grid, update, steps)