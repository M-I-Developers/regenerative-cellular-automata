import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from nn import model
from config import height, width, num_epochs
from update_rule import update

state_grid = torch.zeros((height, width, 16), requires_grad=False)

# plant center seed
seed_height = height // 2
seed_width = width // 2
state_grid[seed_height, seed_width, 3:] = 1.0

lr = 2e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

# for plotting
losses = np.zeros(num_epochs)

# training loop
for epoch in range(num_epochs):
    optimizer.zero_grad() # zero gradients at the start of each epoch

    steps = 5
    total_loss = 0

    print(f"\nSum of elements in state_grid: {state_grid.sum()}")

    for step in range(steps):
        print("Starting step", step)
        state_grid, loss = update(state_grid)
        total_loss += loss
    
    # optimization
    total_loss.backward()  # backpropagate through time (accumulate gradients)
    nn.utils.clip_grad_norm_(model.parameters(), 3) # clip the gradients so they don't explode
    optimizer.step()  # update the grid parameters
    
    losses[epoch] = total_loss.item() / steps

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}")

# plotting the losses
plt.plot(np.arange(1, num_epochs + 1), losses)

plt.title(f"Losses for {steps} steps with lr {lr} Adam gradients updating")
plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.show()