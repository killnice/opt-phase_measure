# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:44:39 2020

@author: lenovo
"""

import cv2

import numpy as np

import math
import time
import imageio

def fun_fillter(x,y,x_size,y_size):
    return math.sqrt(x-x_size) +math.sqrt(y-y_size)

def Thin(image,array):
    #细化函数，输入需要细化的图片（经过二值化处理的图片）和映射矩阵array
    #这个函数将根据算法，运算出中心点的对应值
    #原博客 https://www.cnblogs.com/xianglan/archive/2011/01/01/1923779.html
    h,w = image.shape
    iThin = image
    
    for i in range(h):
        print(i)
        for j in range(w):
            if image[i,j] == 0:
                a = [1]*9
                for k in range(3):
                    for l in range(3):
                        #如果3*3矩阵的点不在边界且这些值为零，也就是黑色的点
                        if -1<(i-1+k)<h and -1<(j-1+l)<w and iThin[i-1+k,j-1+l]==0:
                            a[k*3+l] = 0
                sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                #然后根据array表，对ithin的那一点进行赋值。
                iThin[i,j] = array[sum]*255
    return iThin 
    

 
#映射表
array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]


def get_window_size(x,y,x_size,y_size):
    value = fun_fillter(x,y,x_size,y_size)
    if value>25000:
        return 1
    elif  value>20000 and value<=25000:
        return 2
    elif value>15000 and value<= 20000:
        return 3
    elif value>12000 and value<= 15000:
        return 5
    elif value>10000 and value<=12000:
        return 7
    elif value>8000  and value<=10000:
        return 8
    elif value<=8000:
        return 10
    
def create_gif(image_list, gif_name, duration=0.35):

    imageio.mimsave(gif_name, image_list, 'GIF', duration=duration)
    return
img = cv2.imread('int3.bmp',0) #读灰度图
outimg = np.zeros((img.shape[0],img.shape[1],3),dtype = np.float)
binary = np.zeros((img.shape[0],img.shape[1],3),dtype = np.float )
cv2.namedWindow('image')

cv2.createTrackbar("value1", "image", 50, 255, lambda x: None)
cv2.createTrackbar("value2", "image", 1, 255, lambda x: None)

time.sleep(0.1)
timex = 0
img_list = []
while timex<=255:
    img = cv2.resize(cv2.imread('int3.bmp',0) ,(512,512))#读灰度图
    
    #img = cv2.medianBlur(img,5)
    img = cv2.bilateralFilter(img,9,75,75)
    #cv2.imshow('img',img)
    tx = cv2.getTrackbarPos("value1", "image")
    ty = cv2.getTrackbarPos("value2", "image")
    #img = cv2.adaptiveThreshold(img ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #cv2.THRESH_BINARY,15,2)
    
    ret,img = cv2.threshold(img,tx,255,cv2.THRESH_BINARY)  
#   timex += 1
#   img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#   cv2.THRESH_BINARY,11,2)
    
    
    binary = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    
    cv2.imshow("binary",img)
    
    
    kernel = np.ones((3,3),np.uint8)    
    
    img = cv2.erode(img,kernel,iterations =2)
    
    img = cv2.dilate(img,kernel,iterations = 1)
    
    img = cv2.morphologyEx(img , cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('image',cv2.resize(img,(512,512)))
    
    key = cv2.waitKey(1)&0xFF 
    if key==27:
        break
    elif key == ord('q'):
        circle_img = np.zeros((img.shape),dtype = np.float32)
        cv2.circle(circle_img, (img.shape[1]//2,img.shape[0]//2), int((img.shape[1]*0.8)//2), (1, 1, 1), -1) 
        
        bone = Thin(img,array)
        bone = cv2.bitwise_not(bone) * circle_img 

        cv2.imwrite('bone.jpg',bone)

cv2.destroyAllWindows()