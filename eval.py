import torch
import matplotlib.pyplot as plt
from config import height, width
from update_rule import update

'''
    Run this program after saving a trained model.
    The trained model being evaluated can be changed from the config file.
'''

state_grid = torch.zeros((height, width, 16), requires_grad=False)

# plant center seed
seed_height = height // 2
seed_width = width // 2
state_grid[seed_height, seed_width, 3:] = 1.0

# enable interactive mode
plt.ion()
fig, ax = plt.subplots()

img = ax.imshow(state_grid[:, :, :4].detach().numpy(), interpolation='nearest')
ax.set_title("Grid Progress")
ax.axis('off')

steps = 70

for step in range(steps):
    print("Starting step", step)
    state_grid, _ = update(state_grid, train=False)

    img.set_data(state_grid[:, :, :4].detach().numpy()) # redraw the figure
    plt.pause(0.01)  # allow the plot to update

plt.ioff()
plt.show()