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
