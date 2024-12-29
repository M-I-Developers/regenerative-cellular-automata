import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_automata(initial_grid, update_function, steps, tensor_grid, interval=200):
    """
    Dynamically updates and plots a cellular automaton on the same graphic.

    Parameters:
        
    initial_grid (2D array-like): The starting grid for the automaton.
    update_function (callable): A function to compute the next state of the grid.
    steps (int): Number of steps to animate.
    cmap (str): Colormap for the visualization (default is 'binary').
    interval (int): Time interval (ms) between frames.
    """
    # Initialize the figure
    fig, ax = plt.subplots()
    im = ax.imshow(initial_grid[:, :, :4].astype('uint8'), interpolation='nearest')
    ax.set_title("Dynamic Cellular Automaton")

    # Function to update the animation
    grid = initial_grid.copy()  # Create a mutable copy
    total_loss = 0

    def update(frame):
        print("Reached step: ", frame)
        nonlocal grid, total_loss, tensor_grid
        grid, total_loss, tensor_grid = update_function(grid, total_loss)
        im.set_array(grid[:, :, :4].astype('uint8'))
        return [im]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(steps), interval=interval, blit=True)
    plt.show()

    return total_loss