# restructure_digial_rock
通过图片生成三维岩心（各向同性）

利用Indicate Kriging方法，将一张CT图片生成三维的立体数字岩心
主要分为三步：

1、将一张CT图片进行处理，得到变差函数**对应plot_varg.py**

<img src="https://github.com/Liyurun/restructure_digial_rock/blob/master/CT_img.png" width="300" height="300" />

2、利用变差函数生成三维网络**对应restructure.py**

<img src="https://github.com/Liyurun/restructure_digial_rock/blob/master/exponential_fig.png" width="150" height="150" /><img src="https://github.com/Liyurun/restructure_digial_rock/blob/master/gaussian_fig.png" width="150" height="150" /><img src="https://github.com/Liyurun/restructure_digial_rock/blob/master/scatter_fig.png.png" width="150" height="150" />

3、将三维网络生成可视化文件**对应build_tif.py**

<img src="https://github.com/Liyurun/restructure_digial_rock/blob/master/GIF.gif" width="300" height="300" />

