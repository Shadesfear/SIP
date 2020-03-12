#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is is the main file for the group assingment 6

"""

import exer1




def main():
    
    testImageFolder = "./Week 6/"

    saveImageFolder = "./imageResults/"
    
    exer1.exer11(testImageFolder,saveImageFolder)
    exer1.exer12(testImageFolder,saveImageFolder)    
    exer1.exer13(testImageFolder,saveImageFolder)    
    

    print("done")
    


if __name__ == "__main__":
    main()