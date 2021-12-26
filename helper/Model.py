import os
import tensorflow as tf 
from model import yolo_model2

def cnn_model():
    model = yolo_model2()
    model.load_weights("helper/weight_vgg16.h5")
    return model