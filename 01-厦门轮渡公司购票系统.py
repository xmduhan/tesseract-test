# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:08:20 2014

@author: duhan
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import Image
from sklearn.cluster import KMeans
import pandas as pd
from copy import copy
from matplotlib.colors import ColorConverter
import string


#%% 定义处理过程

def imageToDataFrame(iamge):
    pass


def cleanBackGround(image,bgColor=(255,255,255)):
    '''
    将图片背景涂成统一的颜色
    image   要处理的Image对象
    bgColor 背景要涂成的颜色
    '''    
    # 读取像素数据转化为数据集格式
    pixels = image.load()
    data = []
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            rec = (x,y) + copy(pixels[x,y]) 
            data.append(rec)
    df = pd.DataFrame(data)
    df.columns=('x','y','r','g','b')
    
    # 将所有像素数据分成2类(前景/背景)
    estimator = KMeans(init='k-means++',n_clusters=2)
    estimator.fit(df[['r','g','b']])
    df['label'] = estimator.predict(df[['r','g','b']])
    
    # 通过标签区分出前景和背景
    labelCount = df.groupby('label').x.count().order(ascending=False).reset_index()
    bgLabel = labelCount.label[0]
    df['bg'] = df.label.apply(lambda x:True if x == bgLabel else False)
    
    # 先把背景涂成指定颜色
    bg = df[df.bg]
    for i in range(len(bg)):
        row = bg.irow(i)
        pixels[int(row.x),int(row.y)] = bgColor


def splitImage(image):
    '''
    将图片按字符分开
    image    要切片Image对象
    '''
    pixels = image.load()
    result = []
    for i in range(4):
        smallImage = Image.new( 'RGB', (20,20), "black")
        smallPixels = smallImage.load()
        for x in range(20):
            for y in range(20):
               smallPixels[x,y] = pixels[i*20 + x,y]
        result.append(smallImage)
    return result

    
    
#%% 读取图像文件
image = Image.open('D:\\generateCode.jpg')
plt.imshow(image)

#%% 清理背景
cleanBackGround(image)
plt.imshow(image)

#%%
smallImageList = splitImage(image)
#%%
plt.imshow(smallImageList[2])


