import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from config import height, width, num_epochs
from update_rule import update, compute_loss, model

state_grid = np.zeros(shape=(height, width, 16))

# plant center seed
seed_height = height // 2
seed_width = width // 2
state_grid[seed_height, seed_width, 3:] = 1.0

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

# for plotting
losses = np.zeros(num_epochs)

# training loop
for epoch in range(num_epochs):
    optimizer.zero_grad() # zero gradients at the start of each epoch

    # sample random number of steps
    steps = 10
    total_loss = 0.0

    for step in range(steps):
        print("Starting step", step)
        state_grid = update(state_grid)
        step_loss = compute_loss(state_grid)
        total_loss += step_loss.item()

    # normalize loss
    total_loss /= steps

    losses[epoch] = total_loss
    
    # compute the total loss over all time steps
    total_loss_tensor = torch.tensor(total_loss, requires_grad=True)
    total_loss_tensor.backward()  # backpropagate through time (accumulate gradients)
    
    optimizer.step()  # update the grid parameters

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}")

# plotting the losses
plt.plot(np.arange(1, num_epochs + 1), losses)

plt.title(f"Losses for 10 steps with lr {lr} Adam no retain-graph, removed torch.no_grad")
plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.show()