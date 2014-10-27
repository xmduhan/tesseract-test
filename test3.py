# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 09:31:04 2014

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


#%%
image = Image.open('d:\\1.jpg')
plt.imshow(image)


#%% 将像素矩阵转化为数据集
pixels = image.load() 
data = []
for x in range(image.size[0]):
    for y in range(image.size[1]):
        rec = (x,y) + copy(pixels[x,y]) 
        data.append(rec)
df = pd.DataFrame(data)
df.columns=('x','y','r','g','b')

#%% 获取前景像素 
df['bg']=df.apply(lambda rec:True if rec.r==255 and rec.g==255 and rec.b==255 else False, axis=1)
fg = df[~df.bg]

#%% 对前景像素做聚类
estimator = KMeans(init='k-means++',n_clusters=3)
#estimator = KMeans(init='random',n_clusters=2)
estimator.fit(fg[['r','g','b']])
fg['label'] = estimator.predict(fg[['r','g','b']])

#%% 获取噪点数据集
# 这里假设噪声的像素数量少于字符的数量
labelCount = fg.groupby('label').x.count().order(ascending=True).reset_index()
noise = fg[fg.label==labelCount.label[1]]
#%% 将噪声区域涂成背景色
for i in range(len(noise.index)):
    rec = noise.irow(i)
    pixels[int(rec.x),int(rec.y)] = (255,0,0)

#%%
plt.imshow(image)

#%%
image = Image.open('d:\\1.jpg')