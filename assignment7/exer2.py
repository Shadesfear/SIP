#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file to answer question two in group assingment 7
"""
from keras1 import keras_own
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def extract(image, patchSize = 29):
    # fill out values 
    offset = patchSize // 2
    
    # get shape of image
    r,c = image.shape

    # pad with zeros
    im = np.pad(image, offset, mode = 'constant')
    
    # prepare array to return 
    output = np.zeros((r*c,1,patchSize,patchSize), dtype = np.float32)
    
    # outer patch
    for i in range(r):
        for j in range(c):
            # inner patch         
            for k in range(patchSize):
                for l in range(patchSize): 
                    output[i * c + j,0,k,l] = im[i + k,j + l]
    return output

def dice(predictions, thruth):
    r,c = thruth.shape
    
    intersection = np.equal(predictions, thruth)
    intersect = np.sum(intersection)
    
    dice = 2 * intersect/ (r*c*2)

    return dice


def exer2_2():
    
    model = keras_own()
    ## Load the pretrained network
    model.load_weights('Week_7_export/keras.h5') 
    
    # Load the test data 
    test = np.load('test.npz')
    
    x_test = test['x_test']
    y_test = test['y_test']
    
    ### evaluate the model on the training data
    result = model.evaluate(x_test, y_test)
    print("\n##########\nResults loss, accuracy:\n{}".format(result))



def exer2_3():
#    model = keras_own()
#    
#    test_img = io.imread("./Week_7_export/test_images/image/1003_3_image.png")
#    
#    patches = extract(test_img) 
#    
#    predclass = model.predict(patches)
#    bestguees = np.argmax(predclass, axis=1)
#    
#    pred = np.reshape(bestguees,(256,256))
#    np.save("exer23",pred)


    pred = np.load("exer23.npy")
    
    plt.imshow(pred)
    plt.colorbar()
    plt.savefig("./images/exer23.png",dpi=500,bbox_inches="tight")
    plt.close()


def exer2_4():
    groundTruth = io.imread("./Week_7_export/test_images/seg/1003_3_seg.png")
    seg_own = np.load("exer23.npy")
    
    dice1 = dice(seg_own,groundTruth)
    print("dice value:{}".format(dice1))
    pass

if __name__ == "__main__":
    exer2_2()
    exer2_3()
    exer2_4()