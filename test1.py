# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 12:42:20 2014

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

#%% 对前景像素做二次聚类，分成5类(4个字符和干扰符)
fg = df[df.bg==False]
fg.index = range(len(fg))
estimator = KMeans(init='k-means++',n_clusters=5)
estimator.fit(fg)
fg['label'] = estimator.predict(fg)


#%% 尝试擦去其中一个分类看看能否正常识别
codeRange = string.letters + string.digits

def readCode():
    for labelToErase in range(5):        
        for i in range(len(fg)):
            row = fg.irow(i)
            pixels[int(row.x),int(row.y)] = (row.r,row.g,row.b)      
        
        for i in range(len(fg)):
            row = fg.irow(i)
            if row.label == labelToErase:
                pixels[int(row.x),int(row.y)] = (255,255,255)
        #plt.imshow(image)
        
        codeOrigin = pytesseract.image_to_string(image)
        #codeOrigin = pytesseract.image_to_string(image,boxes=True)
        #codeOrigin = pytesseract.image_to_string(image)        
        print 'codeOrigin=%s' % codeOrigin
        code = ''
        for i in codeOrigin:
            if i in codeRange:
                code = code + i
        #print 'code=%s' % code 
        if len(code) == 4:
            return code
    return None


#%%
code = readCode()
if code :
    print "识别成功!验证码为:%s" % code 
else: 
    print "识别失败!"

#code = pytesseract.image_to_string(image)

         
        
    
#%% 读取图片代表的字符
print pytesseract.image_to_string(image)


#%%
a = {}
for i in range(len(fg.label)):
    key = str(fg.label[i])
    a[key] = a.get(key,0) + 1
    print i
a


#%% 为不同标签涂上不同颜色
colorConverter = ColorConverter()
df['color'] = df.label.apply(lambda x:'white' if x == bgLabel else 'black')


#%% 把颜色涂回图表
for x in range(image.size[0]):
    for y in range(image.size[1]):
        color = df[((df.x==x)&(df.y==y))].irow(0).color
        rgbColor = colorConverter.to_rgb(color)
        pixels[x,y] =  tuple([ int(i * 255) for i in rgbColor])
plt.imshow(image)


#%%



#%%
toReplace = {}
for i in labelCount.index[5:]:
#for i in []:
    print i
    toReplace[labelCount.label[i]] = labelCount.irow(0).label    
    
#%%
df.label = df.label.replace(toReplace)
df.groupby('label').x.count() 

#%%


#%%
for i in sdf.index:
    rec = sdf.ix[i]
    pixels[int(rec.x),int(rec.y)] = (1,1,1)
   
plt.imshow(image)
#%%
pixelList = list(image.getdata())
pixelList



#%%
a = {}
for i in range(len(df.label)):
    key = str(df.label[i])
    a[key] = a.get(key,0) + 1
a


#%%



#%%
pdf = pd.DataFrame(pixelList)




#%%


#%%
a = {}
for i in range(len(labels)):
    key = str(labels[i])
    a[key] = a.get(key,0) + 1
a




#%%
pixelArray = np.array(pixelList)
#%%
pixelArray.shape =(image.size[0],image.size[1],3)

#%%
pixelArray





        

#%%        
import Image
img = Image.new( 'RGB', (255,255), "black") # create a new black image
plt.imshow(img)

#%%
pixels = img.load() # create the pixel map
 
for i in range(img.size[0]):    # for every pixel:
    for j in range(img.size[1]):
        pixels[i,j] = (i, j, 100) # set the colour accordingly
 
plt.imshow(img)
        