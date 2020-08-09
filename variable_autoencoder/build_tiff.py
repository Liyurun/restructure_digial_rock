import PIL.Image as Image
import os
import cv2
import numpy as np
from scipy import misc
import tifffile as tf
path = './test_image/rocktrain/rocktrain/'
a = os.chdir(path)
np.random.rand(3,4)
list_files = os.listdir(os.getcwd())
ddd = []
len(list_files)
for i in range(10):
    lis = 'berea-in-{}.png'.format(i)
    dd = cv2.imread(lis,flags = 0)
    ddd.append(dd[np.newaxis,:])
a = 1
a4 = np.concatenate(ddd[:-1])

tf.imsave('abc.tif',a4)
# ddddd = np.array(np.random.rand(20,20)).astype(np.uint8)
# dddd = np.array(ddd)
# cv2.imwrite('abc.tiff',ddd)
# cc = PIL.open('berea.tif')
# cccc = cv2.imread('berea.tif')
# cc = Image.open('berea.tif',mode='r')
# Image.fromarray(ddddd).save('abc.tif')

# for i in range(len(ddd)):
#     dddc = ddd

