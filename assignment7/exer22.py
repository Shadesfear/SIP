#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file to answer question two in group assingment 7
"""
from Week_7_export.keras1 import keras_own
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def exer():
    
    model = keras_own()
    ## Load the pretrained network
    model.load_weights('Week_7_export/keras.h5') 
    
    # Load the test data 
    test = np.load('test.npz')
    
    x_test = test['x_test']
    y_test = test['y_test']
    print(x_test)
    
    print(x_test.shape)
    print(y_test.shape)
    
    ### evaluate the model on the training data
    result = model.evaluate(x_test, y_test)
    print("\n##########\nResults loss, accuracy:\n{}".format(result))

if __name__ == "__main__":
    exer()