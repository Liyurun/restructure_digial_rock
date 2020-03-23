# restructure_digial_rock
通过图片生成三维岩心（各向同性）

利用Indicate Kriging方法，将一张CT图片生成三维的立体数字岩心
主要分为三步：
![训练图片]('https://github.com/Liyurun/restructure_digial_rock/edit/master/CT_img.png')
1、将一张CT图片进行处理，得到变差函数**对应plot_varg.py**

2、利用变差函数生成三维网络**对应restructure.py**

3、将三维网络生成可视化文件**对应build_tif.py**
