#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is is the main file for the group assingment 6

"""

import exer1, exer2




def main():
    
    testImageFolder = "./Week 6/"

    saveImageFolder = "./imageResults/"
    
    exer1.exer11(testImageFolder,saveImageFolder)
    exer1.exer12(testImageFolder,saveImageFolder)    
    exer1.exer13(testImageFolder,saveImageFolder)    
    
    
    exer2.exer21(testImageFolder,saveImageFolder)
    exer2.exer23iii(testImageFolder,saveImageFolder)
    exer2.exer23iv(testImageFolder,saveImageFolder)
    exer2.exer24(testImageFolder,saveImageFolder)
    exer2.exer25iii(testImageFolder,saveImageFolder)
    exer2.exer25iv(testImageFolder,saveImageFolder)

    print("done")
    


if __name__ == "__main__":
    main()