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
    
    tiles = [img[x:x+shape[0],y:y+shape[1]] for x in range(0,img.shape[0],1) 
             for y in range(0,img.shape[1],1)]
    for i,val in enumerate(tiles):
        if np.shape(i) != shape:
            zero_shape = np.zeros(shape)
            zero_shape[:val.shape[0],:val.shape[1]] = tiles[i]
            tiles[i] = zero_shape
        tiles[i] = np.reshape(tiles[i]/255,(1,1,shape[0],shape[1]))
    return tiles

# def dice(im1, im2):

#     im1 = np.asarray(im1).astype(np.bool)
#     im2 = np.asarray(im2).astype(np.bool)

#     if im1.shape != im2.shape:
#         raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

#     im_sum = im1.sum() + im2.sum()

#     # Compute Dice coefficient
#     intersection = np.logical_and(im1, im2)

#     return 2. * intersection.sum() / im_sum

def dice(predictions, thruth):
    r,c = thruth.shape
    
    intersection = np.equal(predictions, thruth)
    intersect = np.sum(intersection)
    
    dice = 2 * intersect/ (r*c*2)

    return dice


def exer2_3():
    model = keras_own()
    
    test_img = io.imread("./Week_7_export/test_images/image/1003_3_image.png")
    
    # patches = makePatches(test_img) 
    
    # pred = np.empty((0,1))
    # count = 0
    # for i in patches:
        
    #     pred = np.append(pred,np.argmax(model.predict(i)))
    #     count += 1
    #     print(count)
    # pred = np.reshape(pred,(256,256))
    
    pred = np.load("exer23.npy")
    
    plt.imshow(pred)
    plt.colorbar()
    plt.savefig("./images/exer23.png",dpi=500,bbox_inches="tight")
    plt.close()


def exer2_4():
    groundTruth = io.imread("./Week_7_export/test_images/seg/1003_3_seg.png")
    seg_own = np.load("exer23.npy")
    
    dice1 = dice(seg_own,groundTruth)
    pass

if __name__ == "__main__":
    exer2_4()