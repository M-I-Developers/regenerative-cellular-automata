import numpy as np
import random
import torch
import torch.optim as optim
from config import height, width, num_epochs
from update_rule import update
from plot import animate_automata

state_grid = np.zeros(shape=(height, width, 16))

# plant center seed
seed_height = height // 2
seed_width = width // 2
state_grid[seed_height, seed_width, 3:] = 1

tensor_state_grid = torch.tensor(state_grid, dtype=torch.float32)
optimizer = optim.Adam([tensor_state_grid], lr=0.01)

# training loop
for epoch in range(num_epochs):
    optimizer.zero_grad() # zero gradients at the start of each epoch

    # sample random number of steps
    steps = random.randint(5, 10)
    total_loss = animate_automata(state_grid, update, steps, tensor_state_grid)
    
    # compute the total loss over all time steps
    total_loss.backward()  # backpropagate through time (accumulate gradients)
    
    optimizer.step()  # update the grid parameters

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}')
