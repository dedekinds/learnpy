'''
Python科学计算(一)
https://www.shiyanlou.com/courses/348

本课程介绍 python 科学计算，
通过本课程我们可以学习到 Numpy 多维数组，SciPy 科学计算库，
matplotlib 绘图，Sympy 代数系统，用 C 编写 Python 包，Python 并行计算。
'''
1.3 实验目录

    2.1 创建 numpy 数组
        2.1.1 列表生成numpy数组
        2.1.2 使用数组生成函数
        arange
        linspace 与 logspace
        mgrid
        random data
        diag
        zeros 与 ones
    2.2 文件 I/O 创建数组
        2.2.1 CSV
        2.2.2 Numpy 原生文件类型
    2.3 numpy 数组的常用属性
    2.4 操作数组
        2.4.1 索引
        2.4.2 切片索引
        2.4.3 高级索引（Fancy indexing）


2.1.1 列表生成numpy数组
from numpy import *
v=array([[1,2],[3,4]])
print(v)

list 什么都可以放，是动态的，这里的数组是静态的
M = array([[1, 2], [3, 4]], dtype=complex)  
dtype 的常用值有：int, float, complex, bool, object 等。


2.1.2 使用数组生成函数
——————arange————————
x = arange(0, 10, 1) # arguments: start, stop, step  
>>>[0 1 2 3 4 5 6 7 8 9]

——————linspace————————
x=linspace(0, 10, 5) #和matlab一样
>>>[  0.    2.5   5.    7.5  10. ]

——————logspace————————
x=logspace(0, 10, 4, base=e)#取对数
>>>[  1.00000000e+00   2.80316249e+01   7.85771994e+02   2.20264658e+04]

——————mgrid————————
x, y = mgrid[0:5, 0:5] # similar to meshgrid in MATLAB
x=
[[0 0 0 0 0]
 [1 1 1 1 1]
 [2 2 2 2 2]
 [3 3 3 3 3]
 [4 4 4 4 4]]
 y=
[[0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]]

——————random————————
 # uniform random numbers in [0,1]随机数
random.rand(5,1)#均匀分布 random.randn(5,5)就正态分布

[[ 0.52948615]
 [ 0.03186793]
 [ 0.13022328]
 [ 0.93022896]
 [ 0.87229326]]
a[1][0]
Out[24]: 0.03186792586883691

——————diag————————
# a diagonal matrix
diag([1,2,3])

=> array([[1, 0, 0],
          [0, 2, 0],
          [0, 0, 3]])

# diagonal with offset from the main diagonal
diag([1,2,3], k=1) 

=> array([[0, 1, 0, 0],
          [0, 0, 2, 0],
          [0, 0, 0, 3],
          [0, 0, 0, 0]])

——————zeros————————
zeros((3,3))
=> array([[ 0.,  0.,  0.],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]])

——————ones————————
ones((2,2))
array([[ 1.,  1.],
       [ 1.,  1.]])


2.2 文件 I/O 创建数组
——————CSV————————
用genfromtxt获取
data = genfromtxt('test.dat')#读取dat文件
data.shape


用savetxt保存
M = random.rand(3,3)
savetxt("random-matrix.csv", M, fmt='%.5f')


————————索引——————————
from numpy import *
v=random.rand(1,3)
array([[ 0.38496169,  0.14293997,  0.55449541]])

>>>v[0,0]=0.38496169
>>>v[0][0]=0.38496169

取行列和MATLAB一样，M[:,1],注意所有都是从0开始的

M[1,:] = 0表示第二行全部赋予0

A = array([  [n+m*10 for n in range(5)] for m in range(5)  ])
array([[ 0,  1,  2,  3,  4],
       [10, 11, 12, 13, 14],
       [20, 21, 22, 23, 24],
       [30, 31, 32, 33, 34],
       [40, 41, 42, 43, 44]])


————————高级索引——————
row_indices = [1, 2, 3]
A[row_indices]

=> array([[10, 11, 12, 13, 14],
          [20, 21, 22, 23, 24],
          [30, 31, 32, 33, 34]])

col_indices = [1, 2, -1] # remember, index -1 means the last element
A[row_indices, col_indices]

=> array([11, 22, 34])

A[:, col_indices]
=> array([[ 1,  2,  4],
       [11, 12, 14],
       [21, 22, 24],
       [31, 32, 34],
       [41, 42, 44]])

————————索引掩码——————————
B = array([n for n in range(5)])
B
=> array([0, 1, 2, 3, 4])

row_mask = array([True, False, True, False, False])
B[row_mask]
=> array([0, 2])


# same thing
row_mask = array([1,0,1,0,0], dtype=bool)
B[row_mask]
=> array([0, 2])


    2.5 操作 numpy 数组的常用函数
        where
        diag
        take
        choose
    2.6 线性代数
        标量运算
        矩阵代数
        数组/矩阵 变换
        矩阵计算
            矩阵求逆
            行列式
        数据处理
            平均值
            标准差 与 方差
            最小值 与 最大值
            总和, 总乘积 与 对角线和
        对子数组的操作
        对高维数组的操作
    2.7 改变形状与大小
    2.8 增加一个新维度: newaxis
    2.9 叠加与重复数组
        tile 与 repeat
        concatenate
        hstack 与 vstack
    2.10 浅拷贝与深拷贝
    2.11 遍历数组元素
    2.12 矢量化函数
    2.13 数组与条件判断
    2.14 类型转换
    延伸阅读
    三、 实验总结
    License

———————————————
A=array([[ 0,  1,  2,  3,  4],
       [10, 11, 12, 13, 14],
       [20, 21, 22, 23, 24],
       [30, 31, 32, 33, 34],
       [40, 41, 42, 43, 44]])
类似于MATLAB中的find
---A[A>20]
>>>array([21, 22, 23, ..., 42, 43, 44])

---where(A>20)
(array([2, 2, 2, ..., 4, 4, 4], dtype=int64),
 array([1, 2, 3, ..., 2, 3, 4], dtype=int64))

---where(A>20)[0]#[1]的话就是列索引
array([2, 2, 2, ..., 4, 4, 4], dtype=int64)

    np.where()[0] 表示行的索引，
    np.where()[1] 则表示列的索引

---L=where(A>20)
 >>>A[L]=array([21, 22, 23, ..., 42, 43, 44])

——————diag————————
diag(A)
>>>array([ 0, 11, 22, 33, 44])

diag(A,-1)
>>>array([10, 21, 32, 43])

————take函数——————
和高级索引类似，不过也可以用于 list
v2=array([-3, -2, -1,  0,  1,  2])

row_indices = [1, 3, 5]
v2[row_indices] # fancy indexing
=> array([-2,  0,  2])

v2.take(row_indices)
=> array([-2,  0,  2])

用于 list
take([-3, -2, -1,  0,  1,  2], row_indices)
=> array([-2,  0,  2])

————————线性代数——————————
v1=arange(0,5)
A=array([[   0,    1,    4,    9,   16],
       [ 100,  121,  144,  169,  196],
       [ 400,  441,  484,  529,  576],
       [ 900,  961, 1024, 1089, 1156],
       [1600, 1681, 1764, 1849, 1936]])

A*A等价于咩咧里面的点乘

dot(A,v1)表示矩阵乘法

当然实际上A并不是矩阵类型，而是数组类型
真正的矩阵类型是a=matrix(A),此时的*直接表示矩阵的乘法

a.T表示a的转置

——————kron————————
  >>> np.kron([1,10,100], [5,6,7])
  array([  5,   6,   7,  50,  60,  70, 500, 600, 700])
  >>> np.kron([5,6,7], [1,10,100])
  array([  5,  50, 500,   6,  60, 600,   7,  70, 700])

————————实部虚部————————
-real() & imag()
-angle 与 abs 可以分别得到幅角和绝对值

————————求逆和行列式——————————
from scipy.linalg import *#线性代数

-矩阵求逆：
inv(C) # equivalent to C.I 

-行列式：
linalg.det(C)
—————————关于max———————————
m = rand(3,3)
m

=> array([[ 0.09260423,  0.73349712,  0.43306604],
          [ 0.65890098,  0.4972126 ,  0.83049668],
          [ 0.80428551,  0.0817173 ,  0.57833117]])

# global max
m.max()
=> 0.83049668273782951

# max in each column
m.max(axis=0)
=> array([ 0.80428551,  0.73349712,  0.83049668])

# max in each row
m.max(axis=1)
=> array([ 0.73349712,  0.83049668,  0.80428551])

——————reshape————————
用reshape非常快
A=array([ [ n+m*10 for n in range(5)] for m in range(5)])
n, m = A.shape
B = A.reshape((1,n*m))
B[0,0:5] = 5
>>>B=array([[ 5,  5,  5,  5,  5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
           32, 33, 34, 40, 41, 42, 43, 44]])

***但注意到此时A也是发生改变的#浅拷贝B=A，深拷贝的话用copy
>>>A=array([[ 5,  5,  5,  5,  5],
          [10, 11, 12, 13, 14],
          [20, 21, 22, 23, 24],
          [30, 31, 32, 33, 34],
          [40, 41, 42, 43, 44]])

用flatten()的话不会改变A
B = A.flatten()
B

=> array([ 5,  5,  5,  5,  5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31,
          32, 33, 34, 40, 41, 42, 43, 44])

——————————————————

from numpy import *
M=array([[1,2],[3,4]])
for row in M:
    print("row",row)
    for x in row:
        print(x)
#这样相当于是遍历了里面所有的元素
row [1 2]
1
2
row [3 4]
3
4
>>> 

当我们需要遍历数组并且更改元素内容的时候，
可以使用 enumerate 函数同时获取元素与对应的序号：

for row_idx, row in enumerate(M):
    print("row_idx", row_idx, "row", row)

    for col_idx, element in enumerate(row):
        print("col_idx", col_idx, "element", element)

        # 平方每一个元素
        M[row_idx, col_idx] = element ** 2

>>>
row_idx 0 row [1 2]
col_idx 0 element 1
col_idx 1 element 2
row_idx 1 row [3 4]
col_idx 0 element 3
col_idx 1 element 4

M
>>>
array([[ 1,  4],
       [ 9, 16]])

————————————————函数矢量化————————————————
from numpy import *
def Theta(x):
    if x>=0:
        return 1
    else:
        return 0
A=array([-3,-4,5,-8])

Theta_vec=vectorize(Theta)#还有这种操作，可以的
print(Theta_vec(A))#对“函数”进行矢量化改造

[0 0 1 0]
>>> 










