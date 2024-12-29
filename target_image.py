import numpy as np
import cv2

def load_image(image, height, width):
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) # convert BGR image to RGB
    resized_img = cv2.resize(img, dsize=(width, height))
    
    # add alpha channel 
    alpha = np.full((height, width), 255, dtype=np.uint8)
    rgba_img = np.dstack((resized_img, alpha))

    # add other channels to maintain image shape as expected (height, width, 16)
    other_channels = np.zeros((height, width, 12))
    rgba_img = np.dstack((rgba_img, other_channels))
    
    return rgba_img