# -*- coding: UTF-8 –*-
# 上面这一行是python头文件的声明
# 缺省情况下程序需要用ascii码书写，但如果其中写中文的话，python解释器会报错

#点之间的距离
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)
# [[0 1]
#  [1 0]
#  [2 0]]


# Compute the Euclidean（adj 欧几里得的） distance between all rows of x.
print(pdist(x,'euclidean'))
# [1.41421356 2.23606798 1.        ]
'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
在上面的这一个网址中，是官方文档对pdist函数的解释
Returns  Yndarray
Returns a condensed distance matrix Y. For each i and j (where i<j<m ),
where m is the number of original observations(原型观测). The metric dist(u=X[i], v=X[j]) is computed 
and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2.

上面的返回结果是一个一行三列矩阵，因为三个点之间存在三个距离（两两一距离）
这里第0行第0列的1.41421356实际就是x[0,:](此处即x[0,1])与x[1,:]之间距离，
第0行第1列的2.23606798实际就是x[0,:](此处即x[0,1])与x[2,:]之间距离，
我们下面用一个更多行列的矩阵来显示更直观的结果
'''

u = np.array([[0,0,0],[0,0,1],[0,0,3],[0,0,6]])
print("------------")
print(pdist(u,"euclidean"))
# [1. 3. 6. 2. 5. 3.]
# 可以看出，该函数返回的依然是一个一维矩阵（我们可称之为向量，n维向量就是n维空间中两点的直线）
# 向量的第一个参数是u[0,:]与u[1,:]之间的距离，第三个参数是u[0,:]与最后一行u[3,:]的距离
# 第四个参数为u[1,:]与u[2,:]之间的距离2，最后，第六个参数是倒数两行u[2,:]与u[3,:]之间距离
# 可以看到，该向量的返回结果是按行顺序排列的

# 看上面的结果可能难以理解，因此我们需要squareform函数来帮助我们更直观地看到结果

d = squareform(pdist(x, 'euclidean'))
print(d)
# [[ 0.          1.41421356  2.23606798]  
#  [ 1.41421356  0.          1.        
#  [ 2.23606798  1.          0.        ]]
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],  
# 上面这一句是说，二维空间中，计算的是x第i行的这一个坐标与第j行的这一个坐标的欧氏距离，i与j可相同（距离自然为0） 
# and d is the following array:


#记住二维、三维空间中距离公式

'''
scipy.spatial.distance.squareform(X, force=’no’, checks=True)
用来把一个向量格式的距离向量转换成一个方阵格式的距离矩阵，反之亦然。

函数参数
X: 类型是ndarray
官网上说是 Either a condensed or redundant distance matrix. 谁知道这里说的简洁或者冗余距离矩阵是什么鬼？答案揭晓，我知道：首先输入如果是矩阵的话必须是距离矩阵，距离矩阵的特点是
1. d*d的对称矩阵，这里d表示点的个数；
2. 对称矩阵的主对角线都是0；
另外，如果输入的是距离向量的话，必须满足d * (d-1) / 2.
force: 类型是str，可选
强制做’tovector’ 或者’tomatrix’的转换
checks: 类型是bool, 可选
如果是false，将不会进行对阵的对称性和0对角线的检查。
函数返回值
Y：类型是ndarray
如果输入的是简洁的距离矩阵，将返回冗余矩阵；
如果输入的是冗余的距离矩阵，将返回简洁的距离矩阵。

详细介绍可见原文链接https://blog.csdn.net/counsellor/article/details/79555619

我们只需要知道的是，对于任一个二维的距离向量，将其转变成矩阵时，确保对角线上全为0（本行与自身距离），然后：
d[i, j] is the Euclidean distance between x[i, :] and x[j, :],  
上面这一句是说，二维空间中，计算的是x第i行的这一个坐标与第j行的这一个坐标的欧氏距离，i与j可相同（距离自然为0） 
'''

print(squareform(pdist(u,"euclidean")))
'''
[[0. 1. 3. 6.]
 [1. 0. 2. 5.]
 [3. 2. 0. 3.]
 [6. 5. 3. 0.]]
'''

'''
在数学中，欧几里得距离或欧几里得度量是欧几里得空间中两点间“普通”（即直线）距离。使用这个距离，欧氏空间成为度量空间。
相关联的范数称为欧几里得范数。较早的文献称之为毕达哥拉斯度量。
欧几里得距离指在m维空间中两个点之间的真实距离，或者向量的自然长度（即该点到原点的距离），
在二维和三维空间中的欧氏距离就是两点之间的实际距离。
'''


