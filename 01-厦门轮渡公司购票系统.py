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

#%%
def cleanBackGround(image,bgColor):
    