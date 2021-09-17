# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错
'''
Numpy库的核心是数组对象或ndarray对象（n维数组）。你将使用Numpy数组执行逻辑，统计和傅里叶变换等运算。作为使用Numpy的一部分，你要做的第一件事就是创建Numpy数组。本指南的主要目的是帮助数据科学爱好者了解可用于创建Numpy数组的不同方式。

创建Numpy数组有三种不同的方法：

使用Numpy内部功能函数
从列表等其他Python的结构进行转换
使用特殊的库函数
'''
import numpy as np
array = np.arange(20)
print(array)
# Numpy的数组是可变的，这意味着你可以在初始化数组后更改数组中元素的值。 使用print函数查看数组的内容
array[3] = 100
print(array)

# 注意！与Python列表不同，Numpy数组的内容是同质的！必须类型相同
# 因此，如果你尝试将字符串值分配给数组中的元素，其数据类型为int，则会出现错误。
try:
    array[3] ='Numpy'
except Exception as e:
    #这一句except语句可以捕获除程序退出外的一切异常且不显示类型，因为异常都是从Exception类派生
    print("错误：\n",e)
    print(type(e))#<class 'ValueError'>
'''
在Python中，异常也是对象，可对它进行操作。BaseException是所有内置异常的基类，但用户定义的类并不直接继承BaseException，
所有的异常类都是从Exception继承，且都在exceptions模块中定义。Python自动将所有异常名称放在内建命名空间中，
所以程序不必导入exceptions模块即可使用异常。一旦引发而且没有捕捉SystemExit异常，程序执行就会终止。
详见 https://blog.csdn.net/polyhedronx/article/details/81589196
'''

#创建一个二维数组
array = np.arange(20).reshape(4,5)
print(array)

#创建一个三维数组
array = np.arange(27).reshape(3,3,3)
print(array)


#使用arange函数，你可以创建一个在定义的起始值和结束值之间具有特定序列的数组。
print(np.arange(10,35,3))
#[10 13 16 19 22 25 28 31 34]  从10到35（左闭右开），以3递增


#使用其他numpy函数
#使用zeros函数创建一个填充零的数组。函数的参数为一个元组，表示（矩阵一维长度，二维长度,...），在二维情况下表示行数和列数（或其维数）
print(np.zeros((2,4)))
'''
Parameters:
shape : int or tuple of ints
    Shape of the new array, e.g., (2, 3) or 2.
dtype : data-type, optional
    The desired data-type for the array, e.g., numpy.int8. Default is numpy.float64.
order : {'C', 'F'}, optional, default: 'C'
    Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order
'''

#使用ones函数创建一个填充了1的数组，参数介绍同上
print(np.ones((3,4)))


#Return a new array of given shape and type, without initializing（adj 初始化的） entries（cn 条目）.
#注意！！虽说是empty，但初始内容是随机的，取决于内存的状态，也就是没经过初始化
print(np.empty((3,3)))

#full函数创建一个填充给定值的n * n数组。
print(np.full((2,2), 3))

# eye函数可以创建一个n * n的对角矩阵
# 对角矩阵(diagonal matrix)是一个主对角线之外的元素皆为0的矩阵，常写为diag（a1，a2,...,an) 。
# 创建的矩阵对角线上为1，其他为0。
print(np.eye(3,3))

#函数linspace在指定的时间间隔内返回均匀间隔的数字。 例如，下面的函数返回0到10之间的四个等间距数字(可以是浮点数)
print(np.linspace(0, 10, num=4))
#[ 0.          3.33333333  6.66666667 10.        ]





# 从Python列表转换
#除了使用Numpy函数之外，你还可以直接从Python列表创建数组。将Python列表传递给数组函数以创建Numpy数组：
list = [4,5,6]
print("type(list): ",type(list))#<class 'list'>
print(list)
print("type(np.array(list)): ",type(np.array(list)))#<class 'numpy.ndarray'>
print(np.array(list))


#创建二维数组
print(np.array([(1,2,3), (4,5,6)]))


#使用特殊的库函数
#使用特殊库函数来创建数组。例如，要创建一个填充0到1之间随机值的数组，请使用random函数。这对于需要随机状态才能开始的问题特别有用。
print(np.random.random((2,2)))