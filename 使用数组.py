# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错
import numpy as np

# Basic Operators 基本操作符
a = np.arange(25)       #创建一个从0到24，共计25个，逐个递增的一维数组
print("a:\n",a)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]

a = a.reshape((5, 5))       #将该数组调整
print("after reshape operation, now a is\n",a)
'''
 [[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
'''

b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78])
b = b.reshape((5,5))
print("after reshape operation, now b is \n",b)
'''
 [[10 62  1 14  2]
 [56 79  2  1 45]
 [ 4 92  5 55 63]
 [43 35  6 53 24]
 [56  3 56 44 78]]
'''

print(a + b)
'''
[[ 10  63   3  17   6]
 [ 61  85   9   9  54]
 [ 14 103  17  68  77]
 [ 58  51  23  71  43]
 [ 76  24  78  67 102]]
'''
print(a - b)
print(a * b)
print(a / b)

print(a ** 2)
''' 矩阵上各位置的数平方运算
[[  0   1   4   9  16]
 [ 25  36  49  64  81]
 [100 121 144 169 196]
 [225 256 289 324 361]
 [400 441 484 529 576]]
'''
print(a < b) 
''' 矩阵上个位置的数进行比较，输出bool类型
[[ True  True False  True False]
 [ True  True False False  True]
 [False  True False  True  True]
 [ True  True False  True  True]
 [ True False  True  True  True]]
'''
print(a > b)
'''
除了 dot() 之外，上述这些操作符都是对数组进行逐元素运算
'''

print(a.dot(b))
'''
dot() 函数计算两个数组的点积。它返回的是一个标量（只有大小没有方向的一个值）而不是数组。
'''

