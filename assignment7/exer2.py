#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file to answer question two in group assingment 7
"""
from Week_7_export.keras1 import keras_own
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def makePatches(img,shape=(29,29)):
    
    M,N = shape
    
    tiles = [img[x:x+shape[0],y:y+shape[1]] for x in range(0,img.shape[0],1) 
             for y in range(0,img.shape[1],1)]
    for i,val in enumerate(tiles):
        if np.shape(i) != shape:
            zero_shape = np.zeros(shape)
            zero_shape[:val.shape[0],:val.shape[1]] = tiles[i]
            tiles[i] = zero_shape
        tiles[i] = np.reshape(tiles[i]/255,(1,1,29,29))
    return tiles

def exer2_3():
    model = keras_own()
    
    test_img = io.imread("./Week_7_export/test_images/image/1003_3_image.png")
    
    patches = makePatches(test_img) 
    
    pred = np.empty((0,1))
    count = 0
    for i in patches:
        
        pred = np.append(pred,np.argmax(model.predict(i)))
        count += 1
        print(count)
    np.reshape(pred,(256,256))
    pass
    
if __name__ == "__main__":
    exer2_3()