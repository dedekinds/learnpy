#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:44:23 2018

@author: dedekinds
"""

import numpy as np
import cv2
import os

img = cv2.imread(os.getcwd()+'/test.jpg',1)#*cv2.imread(path+0/1)  0：gray 1:color
cv2.imshow('image',img)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()

cv2.imwrite('abv2.png',img)

#_____________________________________
显示彩色图和灰度图

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('test.jpg',1)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # hide x-axes and y-axes
plt.show()

img = cv2.imread('test.jpg',1)#*cv2.imread(path+0/1)  0：gray 1:color
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])#b,g,r--->r,g,b

plt.subplot(121)
plt.xticks([]), plt.yticks([])
plt.imshow(img)

plt.subplot(122)
plt.xticks([]), plt.yticks([])
plt.imshow(img2)
plt.show()
#_____________________________________
自带相机显示(灰色)

import numpy as np
import cv2

cap = cv2.VideoCapture(0)#*cv2.VideoCapture(index)   index=0--->self.camera
while True:
    ret ,frame=cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

————————————————————————————————————————————

#_______________________________________
自带相机显示(彩色)

import numpy as np 
import cv2
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # 从摄像头读取一帧，ret是表明成功与否
    ret, frame = cap.read() 
    if ret:
        cv2.imshow('frame',frame)
    else:
        break
    # 监听键盘，按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
##释放
cap.release()
cv2.destroyAllWindows()
#_____________________________________
摄像头保存视频
import numpy as np 
import cv2
cap = cv2.VideoCapture(0)
#视频编码格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#VideoWriter对象
out = cv2.VideoWriter('cam.avi',fourcc,20.0,(640,480))
while(cap.isOpened()):
    # 从摄像头读取一帧，ret是表明成功与否
    ret, frame = cap.read() 
    if ret:
        #处理得到的帧，然后保存
        #frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
    else:
        break
    # 监听键盘，按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
##释放
cap.release()
out.release()
cv2.destroyAllWindows()

#_______________________________________
读取视频
import numpy as np 
import cv2
cap = cv2.VideoCapture('input.avi')#你应该确保你已经装了合适版本的 ffmpeg, pip install opencv-python
while(cap.isOpened()):
    # 从摄像头读取一帧，ret是表明成功与否
    ret, frame = cap.read() 
    if ret:
        cv2.imshow('frame',frame)
    else:
        break
    # 监听键盘，按下q键退出
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break 
##释放
cap.release()
cv2.destroyAllWindows()

#————————————————————————————————————————————
画矩形

import cv2

img = cv2.imread('test.jpg')
cv2.rectangle(img,(20,50),(60,40),(0,0,0,),3)
cv2.imshow('image',img)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()


