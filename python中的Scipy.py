# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错

from scipy.misc import imread, imresize
import imageio
from PIL import Image
import numpy 
'''
直接使用pip安装misc会报错 ImportError:cannot import name 'imread' from 'scipy.misc'
原因在于包的版本问题，新版本的scipy.misc没有这一个方法
为了对照文档，我们卸载并重新安装：pip3 install scipy==1.2.0
但是依然无法解决问题，发现是我们下错版本了，要用1.0的才行，不过我们不会管这个问题了，
因为我们使用imageio和PIL来作为解决方案，详见我的个人博客
'''

# Read an JPEG image into a numpy array
img = imageio.imread('assets/cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
#img_tinted = imresize(img_tinted, (300, 300))
img_tinted = numpy.array(Image.fromarray(img).resize((300,300)))
#成功将图片保存

# Write the tinted image back to disk
imageio.imwrite('assets/cat_tinted.jpg', img_tinted)
