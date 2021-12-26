import numpy as np

from Model import cnn_model
from Grid import Grid_to_Ann
from Evaluation import NMS, Accuracy
from ShowImage import show_image

def yolo_predict_path(data):
    model = cnn_model()
    size_image = data.shape[1]
    grid  = 3

    predicted = model.predict(data)
    result, prob = Grid_to_Ann(predicted, size_image, grid)
    box  = NMS(result, prob)
    path = show_image(box, grid, data,None,'media/')

    return path

def yolo_predict_image(data):
    model = cnn_model()
    size_image = data.shape[1]
    grid  = 3

    predicted = model.predict(data)
    result, prob = Grid_to_Ann(predicted, size_image, grid)
    box  = NMS(result, prob)

    return box