# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:27:10 2021

@author: bit_YUAN
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlsxwriter
class opt_phase_measure:
    #光电测试大作业 五步移相法+解包裹+三维显示
    def __init__(self):
        self.img_list = []
        for i in range(1,6): #读五张图
            img = cv2.imread(str(i)+'.png',0)
#            img = cv2.imread('int'+str(i)+'.bmp',0)
            img = cv2.bilateralFilter(img,12,60,60) #双边滤波 
            self.img_list.append(img)
        
        self.height = img.shape[1]
        self.width = img.shape[0] #存储干涉图像尺寸
        self.circle_img = np.zeros((img.shape),dtype = np.float32)
        
        cv2.circle(self.circle_img, (self.height//2,self.width//2), int((self.height*0.9)//2), (1, 1, 1), -1)  
        print(self.circle_img.shape)
        
        self.org_phase = np.zeros((self.width ,self.height),dtype = np.float64)
        self.I = np.zeros((len(self.img_list)),dtype = np.float64)
    def zernike(self):
        pass
    
    def fake_color(self,img):
        #输出伪彩色图
        img_real = img.copy()
        fake_color = cv2.applyColorMap(cv2.convertScaleAbs(img_real,alpha=15),cv2.COLORMAP_JET)
        return fake_color
    def phase_warp(self,is_mat): 
        #五步移相法测相位
        #输入 is_mat为1时，使用缓存的数据进行分析 一般输入is_mat为0 
        if not is_mat:
            I = np.zeros((len(self.img_list)),dtype = np.float64)
            for y in range(self.height):
                print('计算进度'+str(int(y/self.height*100))+'%')
                for x in range(self.width):
                    for i in range(len(self.img_list)):
                        I[i] =  self.img_list[i][x,y]
  
                    self.org_phase[x][y] = math.atan2( 2 * (I[3] - I[1]) , (I[0] - 2 * I[2] + I[4] ) )
                    
            np.save('org_phase'+str(self.width)+'.npy',self.org_phase)     
            
        else:
             self.org_phase = np.load('org_phase'+str(self.width)+'.npy')
             x = np.arange(self.width)
             y = np.arange(self.height)
             X,Y = np.meshgrid(x,y)                          
             fig = plt.figure('wrap')
             fig3D = fig.gca(projection='3d')
     
             fig3D.plot_surface(X,Y,-self.org_phase,cmap=plt.cm.rainbow)
             fig3D.view_init(60,30)
    
             plt.show()          
    def phase_unwarp(self): 
         #相位解包裹
         #self.org_phase[np.where(np.isnan(self.org_phase))] = 0
         self.unwrap = self.org_phase
         #self.unwrap[np.where(np.isnan(self.org_phase))] = 255
         midx = self.height//2-1
         midy = self.width//2-1

         self.unwrap[midy, :midx] = np.unwrap(self.unwrap[midy, :midx])
         self.unwrap[midy, midx:] = np.unwrap(self.unwrap[midy, midx:][::-1])
         self.unwrap[midy:   , : ]  = np.unwrap(self.unwrap[midy: , : ])
         self.unwrap[:midy   , : ]  = np.unwrap(self.unwrap[:midy , : ])
         
         self.unwrap = self.unwrap * self.circle_img #去除边界影响 
         print(np.where(self.unwrap == np.min(self.unwrap)))
         x = np.arange(self.width)
         y = np.arange(self.height)
         X,Y = np.meshgrid(x,y)
         
#        self.unwrap = cv2.bitwise_and(self.unwrap,self.circle_img,mask = mask)
         
         fig = plt.figure('unwrap')
         
         fig3D = fig.gca(projection='3d')
 
         fig3D.plot_surface(X,Y,-self.unwrap,cmap=plt.cm.rainbow)
         fig3D.view_init(60,30)

         plt.show()
         fig.savefig('unwrap.png',dpi = 200)
         rangeval = np.max(self.unwrap) - np.min(self.unwrap)
         print('rangeval is ',rangeval)
         self.write_excel(-self.unwrap)

         #使用matlab生成干涉图
    def write_excel(self,data):
        #输出excel
        workbook = xlsxwriter.Workbook('lens_phase.xlsx')     #创建工作簿
        worksheet = workbook.add_worksheet()            #创建工作表
        [row, col] = data.shape
        for i in range(row):
            for j in range(col):
                worksheet.write(i, j,  data[i,j])
        workbook.close()
        print('excel写入完毕')

phase_measure = opt_phase_measure()

phase_measure.phase_warp(1)

phase_measure.phase_unwarp()

cv2.destroyAllWindows()
