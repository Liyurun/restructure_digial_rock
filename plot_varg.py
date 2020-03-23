#!/usr/bin/env python
# coding: utf-8

# In[87]:


from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import pandas as pd
import dill
import time 
import warnings
warnings.filterwarnings("ignore")

# In[98]:


def tiffread(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)
def get_threshold(im,T):
    im[im<T] = 0
    im[im>=T] = 1

def spherical_model(data, X, Y,label="gold_grade"):
  nugget=Y[0]
  sill=(data[label]).var()
  a=range_xy(sill,X,Y)
  Z=pd.DataFrame(np.array(range(0, X[-1])))
  Z[(Z.index<=a)]=nugget+(sill-nugget)*(1.5*Z/a-0.5*np.power(Z/a,3))
  Z[(Z.index>a)]=sill
  return np.array(Z)
    
def exponential_model(data, X, Y,label="gold_grade"):
  nugget=Y[0]
  sill=(data[label]).var()
  a=range_xy(sill,X,Y)
  Z=pd.DataFrame(np.array(range(0, X[-1])))
  Z=nugget+(sill-nugget)*(1-np.exp(-3*Z/a))
  return np.array(Z)
  
  
def gaussian_model(data, X, Y,label="gold_grade"):
  nugget=Y[0]
  sill=(data[label]).var()
  a=range_xy(sill,X,Y)
  Z=pd.DataFrame(np.array(range(0, X[-1])))
  Z=nugget+(sill-nugget)*(1-np.exp(-3*np.square(Z)/np.square(a)))
  return np.array(Z)


def range_xy(sill, X, Y):
  percent95 = 0.95*sill
  for i in range(len(X)):
    if(Y[i] >= percent95):
      return X[i]
  return X[-1]
  


def variogram(data, X = "X", Y = "Y", G="gold_grade", min_angle=0, angle_tolerance=90, no_of_lags = 10, lag_spacing=10, lag_tolerance=5):
    timed = time.time()
    g_values = pd.DataFrame({'distance':[] , 'values':[]})
    x_y_values = defaultdict(lambda:0)
    count = defaultdict(lambda:0)
    for i in range(data.shape[0]):
        timed = 0
        if(angle_tolerance >= 90):
            var_data = pd.DataFrame(np.array(data[(data.index > i) & (np.sqrt(np.square(data[X]-data[X][i])+np.square(data[Y]-data[Y][i])) <= no_of_lags*lag_spacing)])).rename(index = str, columns={0:X, 1:Y, 2:'values'})
        elif(min_angle+angle_tolerance > 90):
            var_data = pd.DataFrame(np.array(data[(data.index > i) & (~((np.arctan2(data[Y]-data[Y][i], data[X]-data[X][i])*180/np.pi > min_angle+angle_tolerance-180) & (np.arctan2(data[Y]-data[Y][i], data[X]-data[X][i])*180/np.pi < min_angle-angle_tolerance))) & (np.sqrt(np.square(data[X]-data[X][i])+np.square(data[Y]-data[Y][i])) <= no_of_lags*lag_spacing)])).rename(index = str, columns={0:X, 1:Y, 2:'values'})
        elif(min_angle-angle_tolerance < -90):
            var_data = pd.DataFrame(np.array(data[(data.index > i) & (~((np.arctan2(data[Y]-data[Y][i], data[X]-data[X][i])*180/np.pi > min_angle+angle_tolerance) & (np.arctan2(data[Y]-data[Y][i], data[X]-data[X][i])*180/np.pi < 180+min_angle-angle_tolerance))) &(np.sqrt(np.square(data[X]-data[X][i])+np.square(data[Y]-data[Y][i])) <= no_of_lags*lag_spacing)])).rename(index = str, columns={0:X, 1:Y, 2:'values'})
        else:
            var_data = pd.DataFrame(np.array(data[(data.index > i) & ((np.arctan2(data[Y]-data[Y][i], data[X]-data[X][i])*180/np.pi >= min_angle-angle_tolerance) & (np.arctan2(data[Y]-data[Y][i], data[X]-data[X][i])*180/np.pi <= min_angle+angle_tolerance)) & (np.sqrt(np.square(data[X]-data[X][i])+np.square(data[Y]-data[Y][i])) <= no_of_lags*lag_spacing)])).rename(index = str, columns={0:X, 1:Y, 2:'values'})
        var_data[X] -= data[X][i]
        var_data[Y] -= data[Y][i]
        var_data["distance"] = np.linalg.norm(var_data[[X, Y]], axis=1)
        var_data = var_data.drop(columns=[X, Y])
        var_data["values"] = var_data["values"] - data[G][i]
        g_values = g_values.append(var_data, ignore_index = True)
        if(i%100 == 0):
            for j in range(0, no_of_lags*lag_spacing+1, lag_spacing):
                range_array = np.array(g_values[(g_values['distance']>j-lag_tolerance) & (g_values['distance'] <= j+lag_tolerance)]['values'])
                x_y_values[j] += np.sum(np.square(range_array), axis = 0)
                count[j] += range_array.shape[0]
            g_values = pd.DataFrame({'distance':[] , 'values':[]}) 
        if i%1000 == 0:
            print('the ',i,' th, time used', time.time() - timed)
            timed = time.time()
    x = []
    y = []
    for keys in x_y_values:
        x.append(keys)
        y.append(x_y_values[keys]/(2*count[keys]))
    return(x, y)


# In[118]:


im=Image.open('CT_img.png')

imarray_temp = np.array(im)
r, g, b = imarray_temp[:,:,0], imarray_temp[:,:,1], imarray_temp[:,:,2]
imarray = 0.2989 * r + 0.5870 * g + 0.1140 * b
get_threshold(imarray,100)
g_values = pd.DataFrame({'X':[] , 'Y':[],'gold_grade':[]})
#捋直
x_values = []
y_values = []
real_values = []
for i in range(0,len(imarray)):
    for j in range(0,len(imarray[0])):
        x_values.append(i)
        y_values.append(j)
        real_values.append(imarray[i][j])
g_values['X'] = x_values
g_values['Y'] = y_values
g_values['gold_grade'] = real_values
g_values
print('start to cal variogram')
[X,Y] = variogram(g_values)
dill.dump_session('img1.pkl')

plt.figure()
plt.scatter(X,Y)
plt.plot(spherical_model(g_values, X, Y), 'g-')
plt.xlabel("Distance")
plt.ylabel("Variogram")
plt.title("Variogram")
plt.savefig('spherical_fig.png')

plt.figure()
plt.scatter(X,Y)
plt.plot(exponential_model(g_values, X, Y), 'b-')
plt.xlabel("Distance")
plt.ylabel("Variogram")
plt.title("Variogram")
plt.savefig('exponential_fig.png')

plt.figure()
plt.scatter(X,Y)
plt.plot(gaussian_model(g_values, X, Y), 'r-')
plt.xlabel("Distance")
plt.ylabel("Variogram")
plt.title("Variogram")
plt.savefig('gaussian_fig.png')