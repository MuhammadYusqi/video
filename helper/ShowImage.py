import cv2
import numpy as np
import os
from django.conf import settings

def grid_box(grid, img):
    img = cv2.resize(img, (300, 300))

    size_grid = img.shape[0] // grid

    for i in range(grid):
        for j in range(grid):
            start = (i*size_grid, j*size_grid)
            end = ((i+1)*size_grid, (j+1)*size_grid)
            color = (255, 0, 0)
            thickness = 1
            box = cv2.rectangle(img, start, end, color, thickness)

    return box
    
def show_image(outputcnn, images, anns=None, save2 = None):
    size_image = 96
    if anns is None:
        anns = np.zeros((images.shape[0],0,4))
    _,h,w,_ = images.shape
    persentase_x = w/size_image
    persentase_y = h/size_image
    for no,(out, img, ann) in enumerate(zip(outputcnn, images, anns)):
        for i in out:
            xmin = int(i[0] * persentase_x)
            xmax = int(i[2] * persentase_x)
            ymin = int(i[1] * persentase_y)
            ymax = int(i[3] * persentase_y)
            start = (xmin, ymin)
            end = (xmax, ymax)
            color = (0, 0, 255)
            thickness = 1
            img = cv2.rectangle(img, start, end, color, thickness)
        for j in ann:
            start = (j[0],j[1])
            end = (j[2], j[3])
            color = (0, 255, 0)
            thickness = 1
            img = cv2.rectangle(img, start, end, color, thickness) 
        # img = grid_box(grid, img)
        if not save2:
            return img
        else:
            img = cv2.resize(img, (300, 300))
            img = (img*255).astype(int)
            path_to_save = os.path.join(settings.BASE_DIR, (save2+str(no)+"-Output.jpg")) 
            cv2.imwrite(path_to_save, img)
            path = save2+str(no)+"-Output.jpg"
            return path
