# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错
import numpy as np

# Boolean masking 
# 布尔屏蔽能让我们根据指定的条件检索数组中的元素
import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a,b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')

plt.show()
'''
利用上述给定的条件来选择图上的不同点。蓝色点(在图中还包括一些绿点，因为绿点把其身下的蓝色点给掩盖了)，
代表值大于0的所有点。绿色点表示值不仅大于0且小于一半π的所有点。
'''

# Incomplete Indexing 缺省索引
a = np.arange(0, 100, 10)
print("a = \n",a)
# [ 0 10 20 30 40 50 60 70 80 90]

b = a[:5]
c = a[a >= 50] #矩阵a中大于等于50的元素
print(b) # >>>[ 0 10 20 30 40]
print(c) # >>>[50 60 70 80 90]
'''
不完全索引是从多维数组的第一个维度获取索引或切片的一种方便方法。例如，
如果数组a=[1，2，3，4，5]，[6，7，8，9，10]，
那么[3]将在数组的第一个维度中给出索引为3的元素，这里是值4。
'''

# Where
a = np.arange(0, 100, 10)
print("in the end , a = \n",a)
# [ 0 10 20 30 40 50 60 70 80 90]

b = np.where(a < 50) 
c = np.where(a >= 50)[0]
print(b) # >>>(array([0, 1, 2, 3, 4], dtype=int64),)
print(c) # >>>[5 6 7 8 9]
'''
where() 函数是另外一个根据条件返回数组中的值的有效方法。
只需要把条件传递给它，它就会返回一个使得条件为真的元素的所在位置的列表。
我们看下面的例子
'''

w = np.array([0,99,89,999,2,3,77])
v = np.where(w >= 77)
print("v:\n",v)
'''
 (array([1, 2, 3, 6], dtype=int64),)
 一维数组w中位置（从0开始）为1,2,3,6的元素满足'>=77'这一个条件
'''

ww = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]])#二维数组
print(ww)
'''
[[1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]]
 一个四行五列的矩阵
'''
vv = np.where(ww > 3)
print("vv:\n",vv)
''' 
(array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=int64), array([3, 4, 2, 3, 4, 1, 
2, 3, 4, 0, 1, 2, 3, 4], dtype=int64))
当where函数中的第一个参数condition是二维矩阵时，
返回的是一个元组tuple，元组的第一个元素是一个array，它的值来源于满足条件的元素的行索引，
两个0行的，三个1行的，四个2行的，五个3行的。
元祖的第二个元素也是一个array，它的值来源于满足条件的元素的列索引。
我们可以依照元组tuple第一个元素array中的值知道总共有多少个符合条件的矩阵元素，然后
根据第二个元素array中的值，在矩阵中依次查找（按行排列）。
总的来说，我们需要用两个数组来确定矩阵中所有满足条件的元素
'''

print(ww[vv])
#[4 5 4 5 6 4 5 6 7 4 5 6 7 8]

b = np.where([[0,3,2],[0,4,0]])
print(b)
#(array([0, 0, 1], dtype=int64), array([1, 2, 1], dtype=int64))
'''
返回的依然是一个元组tuple，第一个元素是一个array，来源于行，第二个元素也是一个array，
来源于列。注意，这一没有删选条件啊，因为where的参数就是b而不是什么，b>1,b>2之类的布尔表达式，
那是怎么回事呢，实际上，它是经过进一步处理了的，将0元素看成是false，将大于0的全部当成是true，
与下面的运行结果是完全等价的。
'''
c = np.array([[False,True,True],[False,True,False]])
print(np.where(c))
#(array([0, 0, 1], dtype=int64), array([1, 2, 1], dtype=int64))

#当参数condition是多维矩阵时的操作

a = [
    [
        [1,2,3],[2,3,4],[3,4,5]
    ],
    [
        [0,1,2],[1,2,3],[2,3,4]
    ]
]
a = np.array(a)
print("a.shape = \n",a.shape)
#形状为  (2, 3, 3)
a_where = np.where(a>3)
print("a_where = \n",a_where)
'''
 (array([0, 0, 0, 1], dtype=int64),
  array([1, 2, 2, 2], dtype=int64), 
  array([2, 1, 2, 2], dtype=int64))
可以看到，返回的是一个元组，第一个元素是array类型，来源于第一个维度满足条件的索引，
第二个元素是array类型，来源于第二个维度满足条件的索引，
第三个元素是array类型，来源于第三个维度满足条件的索引。
'''


'''总结
针对上面的讲述，where的作用就是返回一个数组中满足条件的元素（True）的索引，
且返回值是一个tuple类型，tuple的每一个元素均为一个array类型，array的值即对应某一纬度上的索引。

在之给定一个参数condition的时候，np.where(condition)和condition.nonzero()是完全等价的
————————————————
版权声明：本文为CSDN博主「LoveMIss-Y」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_27825451/article/details/82838402
'''


# 包含三个参数condition，x，y的实际案例
a = np.arange(10)
#[0,1,2,3,4,5,6,7,8,9]
a_where = np.where(a,1,-1)
print(a_where)
#[-1  1  1  1  1  1  1  1  1  1]
#因为只有第一个是0，即false，故而用-1替换了，后面的都大于0，即true，故而用1替换了

a_where_1 = np.where(a>5,1,-1)
print(a_where_1)
#[-1 -1 -1 -1 -1 -1  1  1  1  1]
#前面的6个均为false(小于等于5)，故而用-1替换，后面的四个为true，则用1替换

b = np.where([[True,False],[True,True]],    #第一个参数
    [[1,2],[3,4]],                          #第二个参数
    [[9,8],[7,6]]                           #第三个参数
)
print("b = \n",b)
'''
 [[1 8]
 [3 4]]
'''
'''
第一个True对应的索引位置为（0,0），true在第二个参数中寻找，（0,0）对应于元素1

第二个false对应的索引位置为（0,1），false在第三个参数中寻找，（0,1）对应于元素8

第三个True对应的索引位置为（1,0），true在第二个参数中寻找，（0,0）对应于元素3

第四个True对应的索引位置为（1,1），true在第二个参数中寻找，（0,0）对应于元素4

总结：在使用三个参数的时候，要注意，condition、x、y必须具有相同的维度或者是可以广播成相同的形状，否则会报错，
它返回的结果是一个列表，同原始的condition具有相同的维度和形状。

总结：通过上面的讲述，已经了解到了np.where函数的强大之处，它的本质上就是选择操作，但是如果我们自己编写条件运算，
使用if-else或者是列表表达式这样的语句，效率低下，故而推荐使用np.where。

'''