import numpy as np
import cv2
from PIL import Image
from config import image, height, width

def load_image():
    img = cv2.imread(image)
    resized_img = cv2.resize(img, dsize=(width, height))
    
    # add alpha channel 
    alpha = np.ones((height, width), dtype=np.uint8)
    rgba_img = np.dstack((resized_img, alpha))

    # add other channels to maintain image shape as expected (height, width, 16)
    other_channels = np.zeros((height, width, 12))
    rgba_img = np.dstack((rgba_img, other_channels))
    print(rgba_img.shape)
    
    return rgba_img