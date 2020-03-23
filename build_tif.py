import dill
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

dill.load_session('pic.pkl')
len(dict)
list_l = list()
for i,j in dict.items():
    jT = j.swapaxes(0,2)
    jT = jT[0:-1]
    jT_gray = 0.2989 * jT[0] + 0.5870 * jT[1] + 0.1140 * jT[2]
    list_l.append(jT_gray[np.newaxis, :])
a4 = np.concatenate(list_l)
print(a4.shape)
tf.imsave('temp1.tif', a4)