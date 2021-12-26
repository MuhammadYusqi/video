import numpy as np
import cv2 

def create_data(path=None, image=None):
    size = 96
    if path:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    img_resize = cv2.resize(image, dsize =(size,size), interpolation=cv2.INTER_CUBIC)

    data = np.array(img_resize).reshape(-1, size, size, 3) 
    data = data / 255
    
    return data

