import torch
import torch.nn as nn
from target_image import get_target_image
from config import trained_model_path

# updates network
class NN(nn.Module):
    # The following code operates on a single cell's perception vector
    # (This is why we return to 16 dimension)
    def __init__(self):
        super(NN, self).__init__()
        self.first_layer = nn.Linear(48, 128)
        self.second_layer = nn.Linear(128, 16)
        nn.init.zeros_(self.second_layer.weight) # initialize the weights of the final layer with zero

    def forward(self, perception_vector):
        first_layer_result = nn.functional.relu(self.first_layer(perception_vector))
        second_layer_result = self.second_layer(first_layer_result)
        return second_layer_result
    
def compute_loss(state_grid, updates):
    return mse(state_grid + updates, get_target_image())

mse = nn.MSELoss()
model = NN()

# load the trained model
trained_model = NN()
trained_model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))