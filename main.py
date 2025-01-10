import numpy as np
import random
import torch
import torch.optim as optim
from config import height, width, num_epochs
from update_rule import update, compute_loss, weights
from plot import animate_automata

state_grid = np.zeros(shape=(height, width, 16))

# plant center seed
seed_height = height // 2
seed_width = width // 2
state_grid[seed_height, seed_width, 3:] = 1.0
optimizer = optim.Adam([weights], lr=2e-5)

# training loop
for epoch in range(num_epochs):
    optimizer.zero_grad() # zero gradients at the start of each epoch

    # sample random number of steps
    steps = 10
    total_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)

    for step in range(steps):
        print("Starting step", step)
        state_grid = update(state_grid)

        loss = total_loss + compute_loss(state_grid)
        total_loss = loss
    
    # compute the total loss over all time steps
    total_loss.backward(retain_graph=True)  # backpropagate through time (accumulate gradients)
    
    optimizer.step()  # update the grid parameters

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}')
