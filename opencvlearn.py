#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:44:23 2018

@author: dedekinds
"""

import numpy as np
import cv2
import os

img = cv2.imread(os.getcwd()+'/test.jpg',1)#*
cv2.imshow('image',img)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()

cv2.imwrite('abv2.png',img)

#_____________________________________
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('test.jpg',1)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()

img = cv2.imread('test.jpg',1)#*
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])#*

plt.subplot(121)
plt.xticks([]), plt.yticks([])
plt.imshow(img)

plt.subplot(122)
plt.xticks([]), plt.yticks([])
plt.imshow(img2)
plt.show()
#_____________________________________
import numpy as np
import cv2

cap = cv2.VideoCapture(0)#*
while True:
    ret ,frame=cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






