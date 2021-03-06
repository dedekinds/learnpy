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
————————————SciPy——————————————
SciPy 库建立在 Numpy 库之上，提供了大量科学算法，主要包括这些主题：

    特殊函数 (scipy.special)
    积分 (scipy.integrate)
    最优化 (scipy.optimize)
    插值 (scipy.interpolate)
    傅立叶变换 (scipy.fftpack)
    信号处理 (scipy.signal)
    线性代数 (scipy.linalg)
    稀疏特征值 (scipy.sparse)
    统计 (scipy.stats)
    多维图像处理 (scipy.ndimage)
    文件 IO (scipy.io)


————————数值积分————————————
from scipy.intergrate import quad,dblquad,tplquad#单积分，双重积分，三重积分

数值积分【话说这个误差他是怎么弄出来的？应该是估算的一个上界

from scipy.integrate import quad
def f(x):
    return x
x_lower = 0 # the lower limit of x
x_upper = 1 # the upper limit of x
val, abserr = quad(f, x_lower, x_upper)
print ("integral value =", val, ", absolute error =", abserr )

如果需要输入额外的参数可以用 args 关键字

def integrand(x, n):
    """
    Bessel function of first kind and order n. 
    """
    return jn(n, x)
x_lower = 0  # the lower limit of x
x_upper = 10 # the upper limit of x

val, abserr = quad(integrand, x_lower, x_upper, args=(3,))
print val, abserr 

>>>0.736675137081 9.38925687719e-13

————用lambda表达式
from scipy.integrate import quad
from numpy import *
print(quad(lambda x:exp(-x**2),-Inf,Inf))

(1.7724538509055159, 1.4202636781830878e-08)

——————————————————
二重积分案例
from scipy.integrate import quad
from numpy import *
def integrand(x, y):
    return exp(-x**2-y**2)
x_lower = 0  
x_upper = 10
y_lower = 0
y_upper = 10
val, abserr = dblquad(integrand, x_lower, x_upper, lambda x : y_lower, lambda x: y_upper)
#注意到这里的lambda表达式
print(val)
#注意到我们为y积分的边界传参的方式，这样写是因为y可能是关于x的函数。？


———————————————————————

    2.5 线性代数
        2.5.1 线性方程组
        2.5.2 特征值 与 特征向量
        2.5.3 矩阵运算
        2.5.4 稀疏矩阵
    2.6 最优化
        2.6.1 找到一个最小值
        2.6.2 找到方程的解
    2.7 插值
    2.8 统计学
        2.8.1 统计检验

———————————————————————
        2.5.1 解线性方程：
from scipy.linalg import *
A = array([[1,2,3], [4,5,6], [7,8,9]])
b = array([1,2,3])
x = solve(A, b)#x = solve(A, b.T)，b是否转置没有影响
#直接用数组类型就可以解线性方程，不必matrix

———————————————————————
        2.5.2 特征值 与 特征向量
evals, evecs = eig(A)
evals

=> array([ 1.06633891+0.j        , -0.12420467+0.10106325j,
          -0.12420467-0.10106325j])

evecs
=> array([[ 0.89677688+0.j        , -0.30219843-0.30724366j, -0.30219843+0.30724366j],
          [ 0.35446145+0.j        ,  0.79483507+0.j        ,  0.79483507+0.j        ],
          [ 0.26485526+0.j        , -0.20767208+0.37334563j, -0.20767208-0.37334563j]])
#第k个特征值的特征向量是第k列

————————————稀疏矩阵———————————
from scipy.sparse import *

# dense matrix
M = array([[1,0,0,0], [0,3,0,0], [0,1,1,0], [1,0,0,1]]); M

=> array([[1, 0, 0, 0],
          [0, 3, 0, 0],
          [0, 1, 1, 0],
          [1, 0, 0, 1]])

# convert from dense to sparse
A = csr_matrix(M); A
=> <4x4 sparse matrix of type '<type 'numpy.int64'>'
       with 6 stored elements in Compressed Sparse Row format>

# convert from sparse to dense
A.todense()
=> matrix([[1, 0, 0, 0],
           [0, 3, 0, 0],
           [0, 1, 1, 0],
           [1, 0, 0, 1]])

————填充稀疏矩阵
A = lil_matrix((4,4)) # empty 4x4 sparse matrix
A[0,0] = 1
A[1,1] = 3
A[2,2] = A[2,1] = 1
A[3,3] = A[3,0] = 1
A

=> <4x4 sparse matrix of type '<type 'numpy.float64'>'
       with 6 stored elements in LInked List format>


A.todense()

matrix([[ 1.,  0.,  0.,  0.],
        [ 0.,  3.,  0.,  0.],
        [ 0.,  1.,  1.,  0.],
        [ 1.,  0.,  0.,  1.]])



————————最优化——————————

from scipy import optimize
def f(x):
    return 4*x**3 + (x-2)**2 + x**4
#可以使用 fmin_bfgs 找到函数的最小值：
x_min=optimize.fmin_bfgs(f,-2)
#诡异的迭代方法
print(x_min)

>>>
Optimization terminated successfully.
         Current function value: -3.506641
         Iterations: 5
         Function evaluations: 24
         Gradient evaluations: 8
[-2.67298151]

#也可以直接用这些函数
optimize.brent(f)
=> 0.46961743402759754
optimize.fminbound(f, -4, 2)
=> -2.6729822917513886


——————————函数求根——————————
omega_c = 3.0
def f(omega):
    # a transcendental equation: resonance frequencies of a low-Q SQUID terminated microwave resonator
    return tan(2*pi*omega) - omega_c/omega

optimize.fsolve(f, 0.1)

=> array([ 0.23743014])
#一样要小心迭代法只能求一个根，而案例中tan很多根


——————————————插值————————————
略
from scipy.interpolate import *

————————————统计类————————
from scipy import stats
统计检验
统计分布
等等

——————————matplotlib—————————————
from pylab import *
from numpy import *
x = linspace(0, 5, 10)
y = x ** 2

figure()
plot(x, y, 'r')
xlabel('x')
ylabel('y')
title('title')
show()
#改变边框的大小？

————子图——————
subplot(1,2,1)
plot(x, y, 'r--')
subplot(1,2,2)
plot(y, x, 'g*-');


——————————用fig（类似于句柄
import matplotlib.pyplot as plt
x = linspace(0, 5, 10)
y = x ** 2
fig = plt.figure()

axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')

fig


#在图中插入小子图
import matplotlib.pyplot as plt
x = linspace(0, 5, 10)
y = x ** 2
fig=plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# main figure
axes1.plot(x, y, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')

# insert
axes2.plot(y, x, 'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('insert title');

fig

——————————fig自适应——————————

import matplotlib.pyplot as plt
x = linspace(0, 5, 10)
y = x ** 2


fig, axes = plt.subplots(nrows=1, ncols=2)
#subplots 是布局管理器，自适应fig axes
#h = subplot(m,n,p)/subplot(mnp) 
#将figure划分为m×n块，在第p块创建坐标系，并返回它的句柄
for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

fig.tight_layout()
#标签自适应
fig

————————————图表的尺寸问题————————
fig = plt.figure(figsize=(8,4), dpi=100)
#创建一个800*400像素，每英寸100像素
fig, axes = plt.subplots(figsize=(8,4))

————————————保存图片————————
fig.savefig("filename.png")
%PNG，JPG，EPS，SVG，PGF 和 PDF

——————图例————————
import matplotlib.pyplot as plt
x = linspace(0, 5, 10)
y = x ** 2


fig, ax = plt.subplots()

ax.plot(x, x**2, label="y = x**2")
ax.plot(x, x**3, label="y = x**3")
ax.legend(loc=0) # 图例的位置
#ax.legend(loc=0) # let matplotlib decide the optimal location
#ax.legend(loc=1) # upper right corner
#ax.legend(loc=2) # upper left corner
#ax.legend(loc=3) # lower left corner
#ax.legend(loc=4) # lower right corner
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title');

fig

————————————使用LaTeX，字体大小，字体类型——————————
...latex设置有毛病，一直在弹cmd，不懂

import matplotlib.pyplot as plt
from numpy import *
from matplotlib import rcParams
rcParams.update({'font.size': 18, 'text.usetex': True})#在本机中实现不了
x = linspace(0, 5, 10)
y = x ** 2


fig, ax = plt.subplots()

ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2) # upper left corner
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$y$')
ax.set_title('title');

fig

——————————线型——————————————
线与描点风格

linewidth 或是 lw 参数改变线宽。

linestyle 或是 ls 参数改变线的风格。
具体重新查看即可



——————————————坐标轴的调节————————————
import matplotlib.pyplot as plt
from numpy import *

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, x**2, x, x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x, x**2, x, x**3)
axes[1].axis('tight')#axis('tight') 自动将坐标轴调整的紧凑
axes[1].set_title("tight axes")

axes[2].plot(x, x**2, x, x**3)
axes[2].set_ylim([0, 60])#坐标轴的区域
axes[2].set_xlim([2, 5])
axes[2].set_title("custom axes range");

fig

——————————————————————————————
大招：按照自己的想法设置横纵坐标
import matplotlib.pyplot as plt
from numpy import *

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(x, x**2, x, x**3, lw=2)

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)

yticks = [0, 50, 100, 150]
ax.set_yticks(yticks)
ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18); # use LaTeX formatted labels

fig
#横坐标的是希腊字母
#纵坐标是一些数字

—————————————双坐标——————
fig, ax1 = plt.subplots()

ax1.plot(x, x**2, lw=2, color="blue")
ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
for label in ax1.get_yticklabels():
    label.set_color("blue")

ax2 = ax1.twinx()
ax2.plot(x, x**3, lw=2, color="red")
ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")
for label in ax2.get_yticklabels():
    label.set_color("red")

fig

——————————3D绘图————————————————
#https://www.shiyanlou.com/courses/348/labs/1081/document
    2.3 3D 图
        2.3.1 绘制曲面
        2.3.2 绘制线框
        2.3.3 绘制投影轮廓
        2.3.4 改变视图角度
    2.4 动画
    2.5 后端
        2.5.1 使用 svg 后端生成 svg 图片
        2.5.2 可交互后端

绘制线框型三维图
import matplotlib.pyplot as plt
from numpy import *
from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)
#https://www.shiyanlou.com/courses/348/labs/1081/document


————————————代数系统——————————
from sympy import *
init_printing()#LaTeX输出
x = Symbol('x')

(pi + x)**2

————————————————eval——————————————
from sympy import *
from pylab import *
import numpy
x = Symbol('x')
x_vec = numpy.arange(0, 10, 0.1)
y_vec = numpy.array([N(((x + pi)**2).subs(x, xx)) for xx in x_vec])
fig, ax = subplots()
ax.plot(x_vec, y_vec);

#pi.evalf(n=50) 取pi的50位精确度
#N(y, 5) # same as evalf，保留5位
#y.subs(x,1.5) 表示在y中用1.5来代替x



**特别的，lambdify的方法比上面的数值方法要快得多
from sympy import *
from pylab import *
import numpy
x = Symbol('x')
x_vec = numpy.arange(0, 10, 0.1)
f = lambdify([x], (x + pi)**2, 'numpy')  # 第一个参数是一个列表
                                         # 第二个参数是我们对列表中每个元素的操作
                                         # 第三个参数是我们的
y_vec = f(x_vec)  # now we can directly pass a numpy array and f(x) is efficiently evaluated


时间对比：

1.
from sympy import *#subs
from pylab import *
import numpy
from timeit import timeit  

x = Symbol('x')
#f = lambdify([x], (x + pi)**2, 'numpy')
#y_vec = f(x_vec) 
def func():
   x_vec = numpy.arange(0, 10, 0.1)
   y_vec=numpy.array([N(((x + pi)**2).subs(x, xx)) for xx in x_vec])
   print(y_vec)
t=timeit('func()', 'from __main__ import func', number=1)

>>>0.06101024690926238

2.
from sympy import *#subs
from pylab import *
import numpy
from timeit import timeit  

x = Symbol('x')
#f = lambdify([x], (x + pi)**2, 'numpy')
#y_vec = f(x_vec) 
def func():
   x_vec = numpy.arange(0, 10, 0.1)
   f = lambdify([x], (x + pi)**2, 'numpy')
   y_vec = f(x_vec)
   print(y_vec)
t=timeit('func()', 'from __main__ import func', number=1)

>>>0.00790317583746969#快得多


程序运行时间的代码
#http://www.cnblogs.com/PrettyTom/p/6657984.html
#导入timeit.timeit
from timeit import timeit  

#看执行1000000次x=1的时间：
timeit('x=1')

#看x=1的执行时间，执行1次(number可以省略，默认值为1000000)：
timeit('x=1', number=1)

#看一个列表生成器的执行时间,执行1次：
timeit('[i for i in range(10000)]', number=1)

#看一个列表生成器的执行时间,执行10000次：
timeit('[i for i in range(100) if i%2==0]', number=10000)

—————————————————————
测试一个函数的执行时间：

from timeit import timeit

def func():
    s = 0
    for i in range(1000):
        s += i
    print(s)

# timeit(函数名_字符串，运行环境_字符串，number=运行次数)
t = timeit('func()', 'from __main__ import func', number=1000)
print(t)

    此程序测试函数运行1000次的执行时间


——————————

    由于电脑永远都有其他程序也在占用着资源，你的程序不可能最高效的执行。所以一般都会进行多次试验，取最少的执行时间为真正的执行时间。

from timeit import repeat

def func():
    s = 0
    for i in range(1000):
        s += i

#repeat和timeit用法相似，多了一个repeat参数，表示重复测试的次数(可以不写，默认值为3.)，返回值为一个时间的列表。
t = repeat('func()', 'from __main__ import func', number=100, repeat=5)
print(t) 
print(min(t))

————————————
展开
expand((x+1)*(x+2)*(x+3))
Out[15]: x**3 + 6*x**2 + 11*x + 6

%特别地对三角函数可以这么玩：
expand('sin(a+b)', trig=True)
Out[18]: sin(a)*cos(b) + sin(b)*cos(a)

因式分解
factor(x**3 + 6 * x**2 + 11*x + 6)
Out[19]: (x + 1)*(x + 2)*(x + 3)
