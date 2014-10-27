# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 22:18:51 2014

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


#%% 读取图像文件
image = Image.open('D:\\generateCode.jpg')
plt.imshow(image)

#%% 将数据转化为数据框格式
pixels = image.load() 
data = []
for x in range(image.size[0]):
    for y in range(image.size[1]):
        rec = (x,y) + copy(pixels[x,y]) 
        data.append(rec)
df = pd.DataFrame(data)
df.columns=('x','y','r','g','b')

#%% 将所有像素数据分成2类(前景/背景)
estimator = KMeans(init='k-means++',n_clusters=2)
#estimator = KMeans(init='random',n_clusters=2)
estimator.fit(df[['r','g','b']])
df['label'] = estimator.predict(df[['r','g','b']])

#%% 分出背景颜色
labelCount = df.groupby('label').x.count() 
labelCount.sort(ascending=False)
labelCount = labelCount.reset_index()
labelCount
bgLabel = labelCount.label[0]
df['bg'] = df.label.apply(lambda x:True if x == bgLabel else False)

#%% 先把背景涂成白色
bg = df[df.bg]
for i in range(len(bg)):
    row = bg.irow(i)
    pixels[int(row.x),int(row.y)] = (255,255,255)
plt.imshow(image)


#%% 文字边界(15-19,35-39,55-59,75-79),设置为背景色
borders=range(15,20) + range(35,40) + range(55,60) + range(75,80)
for x in borders:
    for y in range(image.size[1]):
         pixels[int(x),int(y)] = (255,255,255)
plt.imshow(image)

#%% 切割文件到单个字符
img = Image.new( 'RGB', (20,20), "black") # create a new black image
plt.imshow(img)
#%%
px = img.load()
for x in range(20):
    for y in range(20):
       px[x,y] = pixels[x,y]
plt.imshow(img)
img.save('d:\\1.jpg')
#%%

image = Image.open('d:\\1.jpg')
plt.imshow(img)

#%%




#%%
print pytesseract.image_to_string(img,config="-psm 10")

#%% 
