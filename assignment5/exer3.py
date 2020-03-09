#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:16:53 2020

@author: Magnus
"""
import numpy as np
import matplotlib.pyplot as plt
from math import floor

### utility functions

def printf(statement:str):
    if False:
        print(statement)


### 

def exer314():
    windowhalf = 6
    squarehalf = 3 
        
    
    Whitesquare = np.zeros((windowhalf*2 + 1,windowhalf*2 + 1))
    Whitesquare[windowhalf - squarehalf  : windowhalf + squarehalf + 1,windowhalf - squarehalf  : windowhalf + squarehalf + 1] = 1
    
    printf("Centered whitesquare \n{}".format(Whitesquare))
    
    plt.imshow(Whitesquare, cmap='gray')
    plt.savefig("whitesquare")



def translatePoint(Matrix, point):
    """
    translate point with in homogenius coordinates
    """
    
    res = np.dot(Matrix, point)
    
    return res


def nearestNeighborInterpolation(i,j,img, padStrategy):
    
    x = floor(i)
    y = floor(j)

    a = i % 1
    b = j % 1

    offSetX = 0
    offSetY = 0

    if a >= 0.5:
        offSetX = 1
    if b >= 0.5:
        offSetY = 1
    
    return padStrategy(x + offSetX,y + offSetY, img)


def bilinearlyInterpolation(i,j,img, padStrategy):
    
    x = floor(i)
    y = floor(j)

    a = i % 1
    b = j % 1
    
    x0y0 = padStrategy(x,y, img)
    x0y1 = padStrategy(x,y+1, img)
    x1y0 = padStrategy(x+1,y, img)
    x1y1 = padStrategy(x+1,y+1, img)

    return (1-a)*(1-b)*x0y0 + a*(1-b)*x0y1 + (1-a)*b*x1y0 + a*b*x1y1


def zerostrategy(x,y, img):
    r, c = img.shape
    
    
    if 0 <= x and x < r and 0 <= y and y < c:
        return img[x,y]
    else: 
        return 0
        

def translateMatrix(Matrix, img, interpolationStrategy ,paddingstrategy):
    """
    Returns the transformed image.  
    """
    
    r, c = img.shape
    
    printf("(r,c):{},{}".format(r,c))
    
    ## 
    res = np.zeros_like(img)
    
    ## tranform point to new 
    
    for i in range(r):
        for j in range(c):
            
            ## img = matrixInvert res
            tranformationpoint = translatePoint(np.linalg.inv(Matrix), np.array((i,j,1)))
            printf("tranformationpoint : {}".format(tranformationpoint))
            
            x = tranformationpoint[0]
            y = tranformationpoint[1]
            
            imgvalue = interpolationStrategy(x,y,img , paddingstrategy)
            
            res[i, j] = imgvalue
    
    return res

def translate(t, img):
    
    tranformationMatrix = np.identity(3)
    tranformationMatrix[0,2] = t[0]
    tranformationMatrix[1,2] = t[1]
    
    return translateMatrix(tranformationMatrix, img, nearestNeighborInterpolation, zerostrategy)




def Test1():
    
    ## move right   
    windowhalf = 2
    squarehalf = 1 
    
    
    Whitesquare = np.zeros((windowhalf*2 + 1,windowhalf*2 + 1))
    
    res = Whitesquare.copy()
    
    Whitesquare[windowhalf - squarehalf  : windowhalf + squarehalf + 1,windowhalf - squarehalf  : windowhalf + squarehalf + 1] = 1
    
    res[windowhalf - squarehalf  : windowhalf + squarehalf + 1,windowhalf - squarehalf + 1  : windowhalf + squarehalf + 2] = 1
    
    
    plt.imshow(Whitesquare)
    printf("Whitesquare")
    plt.show()
    plt.imshow(res)
    printf("res")
    plt.show()

    tranformationMatrix = np.identity(3)
    tranformationMatrix[1,2] = 1
    
    printf("tranformationMatrix")
    printf(tranformationMatrix)
    
    output = translateMatrix(tranformationMatrix, Whitesquare, nearestNeighborInterpolation, zerostrategy)
    plt.imshow(output)
    printf("res")
    plt.show()

    printf(output)
    printf(res)

    if ((np.equal(output,res).all)):
        printf("True")
    else: 
        printf("False")
   


def exer316():
    
    ## move right   
    windowhalf = 2
    squarehalf = 1 
    
    
    Whitesquare = np.zeros((windowhalf*2 + 1,windowhalf*2 + 1))
    
    res = Whitesquare.copy()
    
    Whitesquare[windowhalf - squarehalf  : windowhalf + squarehalf + 1,windowhalf - squarehalf  : windowhalf + squarehalf + 1] = 1
    
    res[windowhalf - squarehalf  : windowhalf + squarehalf + 1,windowhalf - squarehalf + 1  : windowhalf + squarehalf + 2] = 1
    
    
    plt.imshow(Whitesquare, cmap='gray')
    printf("Whitesquare")
#    plt.show()
#    plt.imshow(res, cmap=gray)
#    printf("res")
    plt.savefig("origwhite")
    plt.show()

    output = translate(np.array((0.6,1.2)), Whitesquare)

    plt.imshow(output, cmap='gray')
    printf("res")
    plt.savefig("transformedwhite")
    plt.show()


def main():
    #Test1()
    exer314()
    exer316()
    printf("done")

if __name__ == "__main__":
    main()