#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is is the main file for the group assingment 6

"""

import exer1, exer2




def main():
    
    testImageFolder = "./Week 6/"

    saveImageFolderexer1 = "./exer1Images/"
    saveImageFolderexer2 = "./exer2Images/"
    
    exer1.exer11(testImageFolder,saveImageFolderexer1)
    exer1.exer12(testImageFolder,saveImageFolderexer1)    
    exer1.exer13(testImageFolder,saveImageFolderexer1)    
    
    
    exer2.exer21(testImageFolder,saveImageFolderexer2)
    exer2.exer23iii(testImageFolder,saveImageFolderexer2)
    exer2.exer23iv(testImageFolder,saveImageFolderexer2)
    exer2.exer24(testImageFolder,saveImageFolderexer2)
    exer2.exer25iii(testImageFolder,saveImageFolderexer2)
    exer2.exer25iv(testImageFolder,saveImageFolderexer2)

    print("done")
    


if __name__ == "__main__":
    main()