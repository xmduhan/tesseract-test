# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:08:20 2014

@author: duhan
"""
#%%
# 导入相关模块
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import Image
from sklearn.cluster import KMeans
import pandas as pd
from copy import copy
from matplotlib.colors import ColorConverter
import string
import os

# 设置当前目录
os.chdir(r"d:\tmp")
srcImageFile = 'generateCode.jpg'
tmpImageFile = 'tmp_image.jpg'
tmpTextFile = 'tmp_text'

#%% 定义处理过程

def imageToDataFrame(image):
    '''
    将一个Image的像素数据转化为对应的数据集形式
    iamge 需要转化的Image对象
    '''    
    #print image.size[0],image.size[1]
    pixels = image.load()
    data = []    
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            rec = (x,y) + copy(pixels[x,y]) 
            data.append(rec)
    df = pd.DataFrame(data)
    df.columns=('x','y','r','g','b')
    return df


def cleanBackGround(image,bgColor=(255,255,255)):
    '''
    将图片背景涂成统一的颜色
    image   要处理的Image对象
    bgColor 背景要涂成的颜色
    '''    
    # 读取像素数据转化为数据集格式
    pixels = image.load()
    df = imageToDataFrame(image)
    
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



def cleanNoise(image,bgColor=(255,255,255),minPixels=60):
    '''
    对前景色进行聚类剔除干扰符
    image   要处理的Image对象
    bgColor 背景要涂成的颜色
    '''    
    pixels = image.load() 
    df = imageToDataFrame(image)
    
    # 获取前景像素 
    df['bg']=df.apply(lambda x:True if x.r==255 and x.g==255 and x.b==255 else False, axis=1)
    fg = df[~df.bg]
    
    # 对前景像素做聚类
    estimator = KMeans(init='k-means++',n_clusters=2)
    estimator.fit(fg[['r','g','b']])
    fg['label'] = estimator.predict(fg[['r','g','b']])
    
    # 获取噪点数据集（这里假设噪声的像素数量少于字符的数量）
    # 排序后干扰符为第1条记录[0]，字符为第2条记录[1]
    labelCount = fg.groupby('label').x.count().order(ascending=True).reset_index()
    # 如果字符所涉及的像素少于minPixels,说明去噪处理错误的将字符进行分割         
    if labelCount.x[1] >= minPixels:          
        noise = fg[fg.label==labelCount.label[0]]
        # 将噪声区域涂成背景色
        for i in range(len(noise.index)):
            rec = noise.irow(i)
            pixels[int(rec.x),int(rec.y)] = bgColor


codeRange = string.uppercase + string.digits

def readFileContent(fileName):
    with open(fileName, 'r') as f:
        fileData = f.read()
    result = ''
    for i in fileData:
        if i in codeRange:
            result = result + i
    return result

def readCharFromImage(image):
    '''
    从图像文件中读取一个字符，这里假设图像文件中仅有一个字符，既在调用tesseract的时
    候使用,使用-psm选项
    image 需要读取的Image对象    
    '''
    image.save(tmpImageFile)    
    os.system('tesseract %s %s -psm 10' % (tmpImageFile,tmpTextFile))
    return readFileContent('tmp_text.txt')   

            
###############################################################################    

#%% 读取图像文件
image = Image.open(srcImageFile)
plt.imshow(image)
# 清理背景
cleanBackGround(image)

#%% 去噪处理
# 将图片切分后去噪声，降低聚类算法出错几率 
smallImageList = splitImage(image)
newImage = Image.new( 'RGB', (80,20), "black")
newPixels = newImage.load()
#%%
for i,smallImage in enumerate(smallImageList):
    cleanNoise(smallImage)
    smallPixels = smallImage.load()
    for x in range(20):
        for y in range(20):
           newPixels[i*20 + x,y] = smallPixels[x,y]

plt.imshow(newImage)

#%% 读取每个切片中的字符，然后进行合并
code = ''.join([readCharFromImage(im) for im in smallImageList])
code


#%%
