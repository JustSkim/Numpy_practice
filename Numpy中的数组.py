# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错
import numpy as np 

my_array = np.array([1, 2, 3, 4, 5]) 
#创建了一个包含5个整数的简单NumPy数组

print("type(my_array) : \n",type(my_array))
#<class 'numpy.ndarray'>
#NumPy提供的最重要的数据结构是一个称为NumPy数组的强大对象。NumPy数组是通常的Python数组的扩展。NumPy数组配备了大量的函数和运算符，可以帮助我们快速编写上面讨论过的各种类型计算的高性能代码

print("my_array:\n",my_array)

print("my_array.shape: \n",my_array.shape)
#打印我们创建的数组的形状：(5, )。意思就是 my_array 是一个包含5个元素的数组。

print("my_array[0] = ",my_array[0])
#打印数组对应位置上的元素

my_array[0] = -1
#修改Numpy数组中的元素

print("Now , my_array is \n",my_array)
#在屏幕上看到：[-1,2,3,4,5]

my_new_array = np.zeros((5)) 
#创建一个长度为5的NumPy数组，但所有元素都为0
print("my_new_array:\n",my_new_array)
# [0. 0. 0. 0. 0.]

my_random_array = np.random.random((5))
#创建一个随机值数组
print("my_random_array:\n",my_random_array)
#[0.56372498 0.81158543 0.25845583 0.03541488 0.7165129 ]

my_2d_array = np.zeros((2, 3)) 
#创建一个二维数组
print("my_2d_array:\n",my_2d_array)
# [[0. 0. 0.]
# [0. 0. 0.]]

my_2d_array_new = np.ones((2, 4))
print(my_2d_array_new)
#[[1. 1. 1. 1.]
#[1. 1. 1. 1.]]

my_array = np.array([[4, 5], [6, 1]])
print("my_array:\n",my_array)
# [[4 5]
# [6 1]]
print("my_array[0][1]:\n",my_array[0][1])
#5
#输出的是索引0行和索引1列中的元素(行列均从0开始)
print("my_array.shape : \n",my_array.shape)
# (2, 2)

my_array_column_2 = my_array[:, 1] 
#使用了冒号(:)而不是行号，而对于列号，我们使用了值1，该操作可以让我们得到特定列
print(my_array_column_2)
#[5 1]