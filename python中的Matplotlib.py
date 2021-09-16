# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错
# matplotlib是一个绘图库，matplotlib.pyplot模块提供了类似于MATLAB的绘图系统
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

# matplotlib中最重要的功能是plot，它允许你绘制2D数据的图像
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
'''
math.pi == np.pi == scipy.pi的判定结果为true，这三个模块都提供pi值的唯一原因是，如果您只使用这三个模块中的一个，
那么您可以方便地访问pi，而不必导入另一个模块。他们没有为pi提供不同的值，np.pi 是一个常数表示圆周率π。
'''
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)  #pyplot.plot是用来画折线图的，里面的参数表示需要的X轴的数据和Y轴的数据
plt.show()  # You must call plt.show() to make graphics appear.

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)


#通过一些额外的工作，我们可以轻松地一次绘制多条线，并添加标题，图例和轴标签：
# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# 也可以使用subplot函数在同一个图中绘制不同的东西
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()




# 使用 imshow 函数显示图像
# 以下例子要借助scipy包的misc模块
img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()