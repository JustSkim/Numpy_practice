# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错
import numpy as np 
a = np.array([[1.0, 2.0], [3.0, 4.0]]) 
b = np.array([[5.0, 6.0], [7.0, 8.0]]) 
sum = a + b   #矩阵相加
difference = a - b #矩阵相减

product = a * b  #注意！乘法运算符执行逐元素乘法（a[0][1]的2与b[0][1]的6相乘得到12）而不是矩阵乘法

quotient = a / b #除法运算也一样，是对应位置上的元素相除即可

print("Sum = \n", sum)
print("Difference = \n", difference) 
print("Product = \n", product)
print("Quotient = \n", quotient) 

# The output will be as follows: 
'''
Sum = [[ 6. 8.] [10. 12.]]
Difference = [[-4. -4.] [-4. -4.]]
Product = [[ 5. 12.] [21. 32.]]
Quotient = [[0.2 0.33333333] [0.42857143 0.5 ]]
'''

matrix_product = a.dot(b) 
#执行矩阵乘法，a.dot(b) 意味着矩阵a左乘矩阵b
print("Matrix Product = ", matrix_product)
#[[19. 22.]
#[43. 50.]]