#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#1
print('hello %s %s %%',%('abx','sss'))#和C语言类似的结构化
print('%s is running'%func.__name__)
————————————————————————
#2
list 语句
classmates=['abc','bcv']
classmates[0]#取出第一个元素
classmates[-1]#最后一个元素
classmates.pop()#删除最后一个元素
classmates.pop(1)#删除指定元素
classmates.append('def')#在最后增加一个元素
classmates.insert(1,'asd')#插入一个元素在某个位置

p = ['asp', 'php']
s = ['python', 'java', p, 'scheme']
s[2][1]→'php'
————————————————————————
#3
tuple 语句
和 list 类似不过 tuple 一旦确定无法更改
t=(1,'a',['a','b'])
t[2][0]='a'

t=(1,)#单个元素时候的定义
————————————————————————
#4判断语句
age = 3
if age >= 18:
    print('adult')
elif age >= 6:#注意是elif
    print('teenager')
else:
    print('kid')
————————————————————————
#p.s.
input 返回的数据类型是 str
需要用 int 处理一下
————————————————————————
#5循环
sum=0
for x in range(101):  #list(range(101)可以得到[0,1,...,100]
    sum=sum+x #range(3,6)表示3:6
print(sum)

or

sum = 0
n = 99
while n > 0:
    sum = sum + n
    n = n - 2
print(sum)

break and continue 是一样的
ctrl+c 强制退出
————————————————————————
使用 dict
>>> d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
>>> d['Michael']
95

d.get('Thomas', -1) 查找某个'Thomas'在不在里面，如果不在就输出-1
 也可以用下面的方法
'Thomas' in d
False

去掉某个人
d.pop('Bob')

————————————————————————
使用 set 实际上就是集合运算
>>> s = set([1, 1, 2, 2, 3, 3])   格式： set(list)
>>> s
{1, 2, 3}

>>>s=set(list(range(3)))
s.add(2)
s.remove(2) 集合运算

s1=set([1,2,3])
s2=set([2,3,4])
s1&s2
s1|s2

————————————————————————
关于可变对象和不可变对象
>>> a = ['c', 'b', 'a']
>>> a.sort()
>>> a
['a', 'b', 'c']


>>> a = 'abc'
>>> a.replace('a', 'A')   str  是不可变对象，但上面的s.add(2) 则直接可以，
'Abc'                     replace 没有改变a 而是创造了一个'Abc'  
>>> a
'abc'


————————————————————————
使用函数

>>> a = abs  变量a指向abs函数
>>> a(-1)  所以也可以通过a调用abs函数！！
1

def my_abs(x):
    if x >= 0:
        return x
    else:
        return -x

from abstest import my_abs 在同一个目录下打开 abstest.py 使用里面 def 的函数my_abs 

def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x  isinstance()是一个用来做参数的类型检查的函数！完善my_abs()

————————————————————————
函数返回多个值 实际上是返回一个 tuple
import math

def move(x, y, step, angle=0): ！！第四个参数表示默认时候是0，即使部署图第四个也行
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny


>>> x, y = move(100, 100, 60, math.pi / 6)
>>> print(x, y)
151.96152422706632 70.0


————————————————————————
默认参数
def enroll(name, gender, age=6, city='Beijing'):
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)

    enroll('Adam', 'M', city='Tianjin') 可以不按照顺序来输入


    一个大坑
def add_end(L=[]):
    L.append('END')
    return L

>>> add_end()
['END']
>>> add_end()
['END', 'END']
>>> add_end()
['END', 'END', 'END']



****这是因为？？怎么理解这个问题呢？？？****重点问题
def add_end(t=[]):
    L=t
    L.append('END')
    return L,t
print(add_end())
print(add_end())
print(add_end())

输出结果是这样，说明L t 相同
(['END'], ['END'])
(['END', 'END'], ['END', 'END'])
(['END', 'END', 'END'], ['END', 'END', 'END'])


首先对于第一个add_end(),此时
t→[]
注意到有第二句话L=t有
L→[]
赋值语句赋予的是地址，所以两个变量L t指向的地址相同
此时修改L.append('END'),那么[]→['END']，L和t指向的地址没有变化
所以L和t都变成了['END']

但是在第二次使用add_end()的时候，难道传入第二次空参数的时候，不是应
该按照"def add_end(t=[]):"中的t=[]，即t再次指向一个[]执行咩？



你如果使用了一个可变类型的默认参数并且调用的时候修改了它，
那么将来的每次调用你都使用的修改之后的对象。
via:http://blog.csdn.net/yangxkl/article/details/44672887


def add_end(L=None):
    if L is None:
        L = []
    L.append('END')
    return L
    上面是正解


————————————————————————
可变参数（一颗星*）
http://www.cnblogs.com/tqsummer/archive/2011/01/25/1944416.html
def funcD(a, b, *c):
  print a
  print b
  print "length of c is: %d " % len(c)
  print c
调用funcD(1, 2, 3, 4, 5, 6)结果是
1
2
length of c is: 4
(3, 4, 5, 6)

其中一个*的时候，传进去后自自动类型变为 tuple
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
print(calc(1, 2))
>>>5

如果已经是一个 list 或者是一个 tuple 呢？
可以采用如下办法
>>> nums = [1, 2, 3]
>>> calc(*nums)
14

完整实例：
def calc(*number):
    sum=0
    for n in number:
        sum=sum+n*n
    return sum
t=list(range(100))
print(calc(*t))

（可变参数的好处是什么？）

————————————————————————
可变参数（两颗星**）
其中一个*的时候，传进去后自自动类型变为 dict

def person(name,age,**kw):
    print('name:',name,'age:',age,'other:',kw)
person('dede',20,tall=181,length=20)

>>> name: dede age: 20 other: {'tall': 181, 'length': 20}


也可以写成
def person(name,age,**kw):
    print('name:',name,'age:',age,'other:',kw)
new={'tall':181,'length':20}
person('dede',20,**new)

对于上面的可变参数还可以设置  关键词参数

(我们希望检查是否有city和job参数)
def person(name, age, **kw):
    if 'city' in kw:
        # 有city参数
        print('有city')
    if 'job' in kw:
        # 有job参数
        print('有job')
person('Jack', 24, city='Beijing', addr='Chaoyang', zipcode=123456)

>>>有city


def person(name, age, *, city, job):
    print(name, age, city, job)
和关键字参数**kw不同，命名关键字参数需要一个特殊分隔符*，
*后面的参数被视为命名关键字参数。

>>> person('Jack', 24, city='Beijing', job='Engineer')
Jack 24 Beijing Engineer


如果函数定义中已经有了一个可变参数，
后面跟着的命名关键字参数就不再需要一个特殊分隔符*了：

def person(name,age,*args,city='beijing',job):
    print(name,age,args,city,job)
person('dede',20,20,30,job='tsinghua')

>>>dede 20 (20, 30) beijing tsinghua


在Python中定义函数，可以用必选参数、默认参数、可变参数、关键字参数和命名关键字参数，这5种参数都可以组合使用。
但是请注意，参数定义的顺序必须是：
必选参数、（普通参数）
默认参数、（city='beijing'）
可变参数、（*）
命名关键字参数、（def person(name, age, *, city, job):）
关键字参数。（**）

def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)

def f2(a, b, c=0, *, d, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)

>>> f1(1, 2)
a = 1 b = 2 c = 0 args = () kw = {}
>>> f1(1, 2, c=3)
a = 1 b = 2 c = 3 args = () kw = {}
>>> f1(1, 2, 3, 'a', 'b')
a = 1 b = 2 c = 3 args = ('a', 'b') kw = {}
>>> f1(1, 2, 3, 'a', 'b', x=99)
a = 1 b = 2 c = 3 args = ('a', 'b') kw = {'x': 99}
>>> f2(1, 2, d=99, ext=None)
a = 1 b = 2 c = 0 d = 99 kw = {'ext': None}


对于任意函数，都可以通过类似
func(*args, **kw)的形式调用它，无论它的参数是如何定义的：
>>> args = (1, 2, 3, 4)
>>> kw = {'d': 99, 'x': '#'}
>>> f1(*args, **kw)
a = 1 b = 2 c = 3 args = (4,) kw = {'d': 99, 'x': '#'}

>>> args = (1, 2, 3)
>>> kw = {'d': 88, 'x': '#'}
>>> f2(*args, **kw)
a = 1 b = 2 c = 3 d = 88 kw = {'x': '#'}


————————————————————————
递归函数
def fact(n):
    if n==1:
        return 1
    return n * fact(n - 1)

汉诺塔问题的递归，好精彩：
def move(n,a,b,c):
    if n==1:
        print(a,'→',c)
    else:
        move(n-1,a,c,b)# 将n-1个盘子由所在地移动到转存地。
        move(1,a,b,c)# 将最底下的盘子移动到目标地。
        move(n-1,b,a,c)# 将余下的n-1个盘子再由转存地移动到目标地。
move(6,'A','B','C')

————————————————————————
切片：
L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']
L[1:3]
>>> ['Sarah', 'Tracy']

>>> L[-2:]
['Bob', 'Jack']
>>> L[-2:-1]
['Bob']
L[0:3]表示，从索引0开始取，直到索引3为止，但不包括索引3


产生1:9的 list:
>>> list(range(1,10))
[1, 2, 3, 4, 5, 6, 7, 8, 9]
或者
L = list(range(100))
print(L[1:10])


前10个数每两个数取一个
>>> L[:10:2]
[0, 2, 4, 6, 8]

全部数，每隔5个取一个
L[::5]

直接复制一个L
L[:]


tuple 也可以直接切片
print((0,1,2,3,4,5,6)[0:3])
或者
L = tuple(range(100))
print(L[0:3])

字符串也可以：（直接切片）
>>> 'ABCDEFG'[:3]
'ABC'
>>> 'ABCDEFG'[::2]
'ACEG'

————————————————————————
Python的蜜汁循环迭代：
对于 dict:
默认是对key循环：
d = {'a': 1, 'b': 2, 'c': 3}
for key in d:
    print(key)

如果要对值循环的话：加上".values"，都循环的话加上".items"
d = {'a': 1, 'b': 2, 'c': 3}
for key in d.values():
    print(key)


对字符串也行：
for ch in 'ABC':
    print(ch)
A
B
C
>>> 

用 enumerate()函数强行建立索引， enumerate(枚举的意思)
for i, value in enumerate(['A', 'B', 'C']):
     print(i, value)
0 A
1 B
2 C
>>> 

蜜汁循环系列：
for x, y in [(1, 1), (2, 4), (3, 9)]:
     print(x, y)
1 1
2 4
3 9
>>> 

————————————————————————
列表生成式
list(range(1, 11))

如果要生成：[1x1, 2x2, 3x3, ..., 10x10]
可以考虑循环：
L = []
for x in range(1, 11):
    L.append(x * x)

或者用列表生成式：****
L=[x*x for x in range(1,11)]
print(L)
还可以增加判断
>>> [x * x for x in range(1, 11) if x % 2 == 0]
[4, 16, 36, 64, 100]

还可以使用两层循环：
>>> [m + n for m in 'ABC' for n in 'XYZ']
['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']

dict的items()可以同时迭代key和value：
d = {'x': 'A', 'y': 'B', 'z': 'C' }
for k, v in d.items():
     print(k, '=', v)
x = A
y = B
z = C
>>> 

其他用法
>>> d = {'x': 'A', 'y': 'B', 'z': 'C' }
>>> [k + '=' + v for k, v in d.items()]
['y=B', 'x=A', 'z=C']

最后把一个list中所有的字符串变成小写：

>>> L = ['Hello', 'World', 'IBM', 'Apple']
>>> [s.lower() for s in L]
['hello', 'world', 'ibm', 'apple']

作业：将L1的字符串变小写（lower对非字符串会报错）
L1 = ['Hello', 'World', 18, 'Apple', None]
L2=[s.lower() for s in L1 if isinstance(s,str)]
print(L2)

————————————————————————
生成器generator
>>> L = [x * x for x in range(10)]
>>> L
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
>>> g = (x * x for x in range(10))
>>> g
<generator object <genexpr> at 0x1022ef630>

节省空间的作法，每次调用next(g)可以出来下一个数
也可以用循环
g=(x*x for x in range(10))
for n in g:
    print(n)

generator的函数，在每次调用next()的时候执行，
遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'
for n in fib(6):
     print(n)


#杨辉三角
def triangles():#用generator
    L = [1]
    while True:
        yield L
        L = [1] + [L[i-1] + L[i] for i in range(len(L)) if i > 0] + [1]
n=0
for t in triangles():
    print(t)
    n=n+1
    if n==10:
        break

自己的一般写法
def triangle(max):
    L=[1]
    n=1
    while n<max:
        print(L)
        L=[1]+[L[i-1]+L[i] for i in range(len(L)) if i>0]+[1]
        n=n+1
triangle(5)

 强行用generator
 def triangle(max):
    L=[1]
    n=1
    while n<max:
        yield(L)
        L=[1]+[L[i-1]+L[i] for i in range(len(L)) if i>0]+[1]
        n=n+1
for k in triangle(5):
    print(k)
————————————————————————

迭代器
你可能会问，为什么 list、dict、str 等数据类型不是Iterator？
他们都是可迭代的（Iterable）

这是因为Python的Iterator对象表示的是一个数据流，
Iterator对象可以被next()函数调用并不断返回下一个数据，
直到没有数据时抛出StopIteration错误。
可以把这个数据流看做是一个有序序列，
但我们却不能提前知道序列的长度，
只能不断通过next()函数实现按需计算下一个数据，
所以Iterator的计算是惰性的，
只有在需要返回下一个数据时它才会计算。

把 list、dict、str 等Iterable变成Iterator可以使用 iter()函数：
>>> isinstance(iter([]), Iterator)
True
>>> isinstance(iter('abc'), Iterator)
True

1.凡是可作用于for循环的对象都是Iterable类型；
2.凡是可作用于next()函数的对象都是Iterator类型，它们表示一个惰性计算的序列；
3.集合数据类型如 list、dict、str 等是Iterable但不是Iterator，不过可以通过iter()函数获得一个Iterator对象。
4.Python的for循环本质上就是通过不断调用next()函数实现的

————————————————————————
函数式编程
允许把函数本身作为参数传入另一个函数，还允许返回一个函数
def add(x,y,f):
    return f(x)+f(y)
print(add(-5,6,abs))
传函数

from math import sqrt
def same(x,*fs):
    s=[f(x) for f in fs]+
    return s
print(same(2,sqrt,abs))

————————————————————————
map/reduce 函数

map(f,Iterable)→::generator（Iterator，也就是一个惰性序列）
 用法格式（所以最后要用一个 list 处理）
def f(x):
    return x*x
r=map(f,list(range(1,10)))
print(list(r))
>>>[1, 4, 9, 16, 25, 36, 49, 64, 81]

再例如：将一串数字变为字符串：
print(list(map(str,list((range(1,10))))))
['1', '2', '3', '4', '5', '6', '7', '8', '9']
>>> 


————————————————————————
reduce 函数
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
要加这句“ from functools import reduce”
实现数列求和：
from functools import reduce
def add(x,y):
    return x+y
print(reduce(add,list(range(1,10))))


把序列[1, 3, 5, 7, 9]变换成整数13579:
from functools import reduce
def add(x,y):
    return x*10+y
L=[1,3,5,7,9]
print(reduce(add,L))

由此写一个str2int函数：
from functools import reduce
def str2int(s):
    def add(x,y):
        return x*10+y
    def char2num(s):#注意到'13546'是可迭代类型
        dicttemp={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        return dicttemp[s]
    return reduce(add,map(char2num,s))
print(str2int('125486'))
>>> 125486

还可以用lambda函数进一步简化成：

from functools import reduce
def char2num(s):
    return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
def str2int(s):
    return reduce(lambda x, y: x * 10 + y, map(char2num, s))

------------------
*习题1：
利用map()函数，把用户输入的不规范的英文名字，变为首字母大写，其他小写的规范名字。
输入：['adam', 'LISA', 'barT']，输出：['Adam', 'Lisa', 'Bart']：
我的笨方法：
def normalize(name):
    Ltemp=name.lower()
    return Ltemp[0].upper()+Ltemp[1:]
# 测试:
L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)

------------------
*习题2：
Python提供的sum()函数可以接受一个list并求和，
请编写一个prod()函数，可以接受一个list并利用reduce()求积
from functools import reduce
def prod(L):
    def prodtemp(x,y):
        return x*y
    return reduce(prodtemp,L)
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))



------------------
*习题3：
利用map和reduce编写一个str2float函数，
把字符串'123.456'转换成浮点数 123.456：
from functools import reduce
def str2float(s):
    def char2num(s):
        #'13.31'→[1,3,-1,3,1]
        dicttemp={'.':-1,'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        return dicttemp[s]
    def add(x,y):
        #跳过'.'
        if y>=0:
            return 10*x+y
        else:
            return x
    def findfloat(liststr):
        #找小数点的位置
        num=0
        for m in liststr:
            if m>=0:
                num=num+1
            else:
                break
        return num
    tempnum=len(s)-findfloat(list(map(char2num,s)))#得到应该除以10的次数
    L=reduce(add,map(char2num,s))
#    print(L)
#    for t in range(1,tempnum):
#        L=L/10
    return L/(10**(tempnum-1))
print(str2float('123.456'))



QAQ好冗长的说：
比较好的解答：
from functools import reduce

def str2float(s):
    XiaoShu = len(s) - s.find('.') - 1      # 计算小数位数
    s = s.replace('.', '')      # 删除小数点
    def char2num(s):
        return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
    num = reduce(lambda x, y: 10 * x + y, map(char2num, s))      # 转换成不带小数的整数
    num = num / (10 ** XiaoShu)      # 移动小数点
    return num

print('str2float(\'123.456\') =', str2float('123.456'))

————————————————————————
filter(f,list)→:generator（Iterator，也就是一个惰性序列）
 用法格式（所以最后要用一个 list 处理）
f作用于 list 上，给出 True 和 False,True 留下

去掉偶数：
def is_odd(n):
    return n % 2 == 1
list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
# 结果: [1, 5, 9, 15]

去掉空白和 None
def not_empty(s):
    return s and s.strip()#strip()可以去掉 str 首尾的空白
L=list(filter(not_empty, ['A', '', 'B', None, 'C', '  ']))
print(L)


————————————————————————
用 filter 筛选素数
def _odd_iter():#生成无穷奇数列
    n = 1
    while True:
        n = n + 2
        yield n
        
def _not_divisible(n):
    return lambda x: x % n > 0#返回一个函数，厉害了大兄弟


def primes():
    yield 2
    it = _odd_iter() # 初始序列
    while True:
        n = next(it) # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it) # 构造新序列

        
# 打印1000以内的素数:
for n in primes():
    if n < 1000:
        print(n)
    else:
        break


习题：请利用filter()滤掉非回数：例如 12321，909
def is_palindrome(n):
    temp=str(int(n))
    test=0
    for x in range(round(len(temp)/2)):
        if temp[x]==temp[-x-1]:
            pass
        else:
            test=1
    if test==1:
        return False
    else:
        return True

output = filter(is_palindrome, range(1, 1000))
print(list(output))


神解答：给跪了：
def is_palindrome(n):
        return str(n)==str(n)[::-1]# [::2]表示每隔2个取一个

output = filter(is_palindrome, range(1, 1000))
print(list(output))

————————————————————————
排序算法 sorted():

按绝对值大小排序
>>> sorted([36, 5, -12, 9, -21], key=abs)
[5, 9, -12, -21, 36]

无视大小写排序（否则大写都在小写前面
sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)
['about', 'bob', 'Credit', 'Zoo']

要进行反向排序，不必改动key函数，可以传入第三个参数reverse=True：
reverse （倒退的意思）
>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True)
['Zoo', 'Credit', 'bob', 'about']
当然也可以这样：
sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.upper)


习题1：
假设我们用一组tuple表示学生名字和成绩：
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
请用sorted()对上述列表分别按名字排序：

L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    return t[0]
L2 = sorted(L, key=by_name)
print(L2)


习题2：
再按成绩从高到低排序：
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    return t[1]
L2 = sorted(L, key=by_name,reverse=True)
print(L2)

————————————————————————
返回一个函数
L=list(range(1,10))
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

f=lazy_sum(*L)
print(f())

我们在函数lazy_sum中又定义了函数sum，并且，
内部函数sum可以引用外部函数lazy_sum的参数和局部变量
当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中，
这种称为“闭包（Closure）”的程序结构拥有极大的威力。


闭包(Closure):内层函数引用了外层函数的变量(包括它的参数),
然后返回内层函数的情况,这就是闭包.

>>> f1 = lazy_sum(1, 3, 5, 7, 9)
>>> f2 = lazy_sum(1, 3, 5, 7, 9)
>>> f1==f2
False
即使输入相同的参数返回的函数都是不同的


————————————————————————
函数闭包
def count():
    fs = []
    for i in range(1, 4):
        def f():
             return i*i
        fs.append(f)
    return fs

f1, f2, f3 = count()
print(f1())
print(f2())
print(f3())

>>> 
9
9
9
结果全是9，这是因为原因就在于返回的函数引用了变量i，但它并非立刻执行。
等到3个函数都返回时，它们所引用的变量i已经变成了3，因此最终结果为9。

返回闭包时牢记的一点就是：
**返回函数不要引用任何循环变量，或者后续会发生变化的变量。**

解决方案：
def count():
    def f(j):
        def g():
            return j*j
        return g
    fs = []
    for i in range(1, 4):
        fs.append(f(i)) # f(i)立刻被执行，因此i的当前值被传入f()
    return fs

————————————————————————
匿名函数
list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))

[x*x for x in range(1,10)]

因为函数没有名字，不必担心函数名冲突:
f=lambda x:x*x

————————————————————————
装饰器：
>>> now.__name__
'now'
获取函数的名字

在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）
def use_logging(func):
    print('%s is running'%func.__name__)
    func()
def bar():
    print('i am bar')
use_logging(bar)

为了增强bar()，实际上是可以直接在bar里面加代码，
但是如果此时bar1(),bar2()都有需求呢？就可以用上面的方法，但是
有木有更好的方式？：装饰器
https://www.zhihu.com/question/26930016

def use_logging(func):
    def wrapper(*args,**kw):
        print('%s is running'%func.__name__)
        return func(*args,**kw)
    return wrapper
    
def bar(sk):
    print('i am bar %s'%sk)

bar=use_logging(bar)
bar('dk')

bar is running
i am bar dk
>>> 


实际上这里的*args,**kw是给bar()传的参数，实在是厉害啊！
可以用下面的@符号（称为语法糖，避免再次赋值
def use_logging(func):
    def wrapper(*args,**kw):
        print('%s is running'%func.__name__)
        return func(*args,**kw)
    return wrapper

@use_logging
def bar():
    print('i am bar')

bar()

如果decorator本身需要传入参数，那就需要编写一个返回decorator的高阶函数
def log(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator

@log('execute')
def now():
    print('2015-3-25')

>>> now()
execute now():
2015-3-25


为了防止某些严重的错误，完整的decorator的写法:
import functools

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper
@log
def bar():
    print('i am bar')

如果装饰器本身需要传进参数，那么：
import functools

def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)#其实这里不一定要返回，直接写func(*args,**kw)即可
        return wrapper
    return decorator
@log('execute')
def bar():
    print('2015-3-25')



习题1：请编写一个decorator，
能在函数调用的前后打印出'begin call'和'end call'的日志。
import functools
def log(func):
    @functools.wraps(func)
    def wrapper(*args,**kw):
        print('begin call')
        func(*args,**kw)
        print('end call')
    return wrapper
@log
def bar():
    print('i am dedekinds')
bar()

习题2：写出一个@log 的decorator，使它既支持：
@log
def f():
    pass
又支持：
@log('execute')
def f():
    pass

import functools
def log(text):
    if isinstance(text,str):#如果是字符串的话
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                print('begin')
                print('%s'%text)
                func(*args,**kw)
                print('end')
            return wrapper
        return decorator
    else:
            @functools.wraps(func)
            def wrapper(*args, **kw):
                print('begin')
                func(*args,**kw)
                print('end')
            return wrapper
@log('sdfdsf')
def bar():
    print('i am dedekinds')
bar()


————————————————————————
偏函数
int('1234',5)
    #5进制'1234'转换为10进制
194
>>> 
为了不每次都输入一波int('str',base)
可以定义一个新的函数，以二进制为例：
def int2(x,base=8):
    return int(x,base)
如今有一种高级用法：

import functools
int2=functools.partial(int,base=8)
print(int2('124525'))
43349
>>> 

————————————————————————
模块
一个例子：
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '
#表示模块的文档注释，任何模块代码的第一个字符串都被视为模块的文档注释；

__author__ = 'Michael Liao'#这个是表示作者
#以下才是重点
import sys
def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':
    #这句话的意思就是，当模块被直接运行时，代码运行
    #以下代码块将被运行，当模块是被导入时，代码块不被运行
    test()

要调用的话只能（if __name__=='__main__':
>>> import hello
>>>hello.test()
Hello, world!



关于 import:
如果把下面的代码保存为 hzz.py

def _private_1(name):#在前面加了_命名的变量无法被引用
    return 'Hello, %s' % name
def _private_2(name):
    return 'Hi, %s' % name
def greeting(name):
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)


用法1： 
import hzz
>>> hzz.greeting('hsdjfhsjfhs')
'Hello, hsdjfhsjfhs'

用法2：
>>> from hzz import greeting
>>> greeting('dfff')
'Hello, dfff'

————————————————————————
使用第三方库
在cmd中直接输入下列命令安装：
pip install Pillow

可以在这里搜库：https://pypi.python.org/pypi

from PIL import Image#安装了Pillow后
im = Image.open('test.png')
print(im.format, im.size, im.mode)
im.thumbnail((200, 100))
im.save('thumb.jpg', 'JPEG')
#生成一个thumb.jpg的缩略图


————————————————————————
面向对象编程


1.面相过程的编程：
std1 = { 'name': 'Michael', 'score': 98 }
std2 = { 'name': 'Bob', 'score': 81 }
def print_score(std):
    print('%s:%s'%(std['name'],std['score']))
print_score(std2)

2.面相对象的编程：
class Student(object):
    def __init__(self,name,score):
        self.name=name
        self.score=score
        
    def print_score(self):
        print('%s %s'%(self.name,self.score))

bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
bart.print_score()
lisa.print_score()

————————————————————————
类（Class）和实例（Instance）

class Student(object):
    pass
- Student类名一般大写
-(object)，表示该类是从哪个类继承下来的


创建实例的时候，把一些我们认为必须绑定的属性强制填写进去:
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

-__init__方法的第一个参数永远是self，表示创建的实例本身

有点像结构体，不过此时可以传入参数比如
bart = Student('Bart Simpson', 59)
>>> bart.name
'Bart Simpson'


class Student(object):
    def __init__(self,name,score):
        self.name=name
        self.score=score
bart=Student('dedekinds',100)
print(bart.name)
print(bart.score)
dedekinds
100
>>> 

注意的是
__init__()
不是
__int__()

输入的参数也可以是类

class Student(object):
    def __init__(self,name,score):
        self.name=name
        self.score=score
bart=Student('dedekinds',100)

def pri(std):
    print('%s %s'%(std.name,std.score))
pri(bart)

dedekinds 100
>>> 


不过我们也可以对上述函数进行封装，如下所示：
class Student(object):
    def __init__(self,name,score):
        self.name=name
        self.score=score
        
    def print_score(self):
        print('%s %s'%(self.name,self.score))
    def get_grade(self):
            if self.score >= 90:
                return 'A'
            elif self.score >= 60:
                return 'B'
            else:
                return 'C'
                
bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)
bart.print_score()
lisa.print_score()
print(bart.get_grade())

Bart Simpson 59
Lisa Simpson 87
C
>>> 

实际上和结构体这种静态的变量不同，类是动态的，比如接着
上面的程序而言，如果我要加入一个新的 age 
lisa.age=7
print(lisa.age)

直接像上面那样就行了（不过怎么知道都加了啥？

————————————————————————
访问限制

理论上上面的变量是可以随意更改的
>>> bart = Student('Bart Simpson', 98)
>>> bart.score
98
>>> bart.score = 59
>>> bart.score
59

为了防止这种现象，在Python中经常在变量前增加两个__就可以变成私有变量
class Student(object):

    def __init__(self, name, score):
        self.__name = name
        self.__score = score

    def print_score(self):
        print('%s: %s' % (self.__name, self.__score))


bart = Student('Bart Simpson', 59)
lisa = Student('Lisa Simpson', 87)

bart.__score=100
bart.print_score()
print(bart.__score)

Bart Simpson: 59
100
>>> 

可以发现更改不了，如果没有__，那么在外部是可以修改的，如上上上所示


如果想让类能返回值
class Student(object):
    ...

    def get_name(self):
        return self.__name

如果想类改变值

class Student(object):
    ...
    def set_score(self,score):
        self.__score=score

P.S.注意的是：
1.__XXX__双下划线是”特殊变量“，
特殊变量是可以直接访问的，不是private变量

2.有些时候，你会看到以一个下划线开头的实例变量名，比如_name，
这样的实例变量外部是可以访问的，但是，按照约定俗成的规定，
当你看到这样的变量时，意思就是，“虽然我可以被访问，但是，
请把我视为私有变量，不要随意访问”。

3.双下划线开头的实例变量是不是一定不能从外部访问呢？其实也不是。
不能直接访问 __name是因为Python解释器对外把__name变量改成
了_Student__name，所以，仍然可以通过_Student__name来访问__name变量：
>>> bart._Student__name
'Bart Simpson'


————————————————————————
继承和多态
先定义一个类：
class Animal(object):
    def run(self):
        print('Animal is running...')

当我们需要编写Dog和Cat类时，就可以直接从Animal类继承：
class Dog(Animal):
    pass
class Cat(Animal):
    pass

dog=Dog()
dog.run()


完整代码如下：类的继承：
class Animal(object):
    def run(self):
        print('Animal is running...')
class Dog(Animal):
    pass
class Cat(Animal):
    pass
dog=Dog()
dog.run()

Animal is running...
>>> 



子类的run()覆盖了父类的run()，修改方便
class Animal(object):
    def run(self):
        print('Animal is running...')
class Dog(Animal):
    def run(self):
        print('Dog is running...')
class Cat(Animal):
    def run(self):
        print('Cat is running...')
dog=Dog()
dog.run()

Dog is running...
>>> 

在代码运行的时候，
总是会调用子类的run()。这样，我们就获得了继承的另一个好处：多态。

关于多态， class 实际上定义的上一个类型
接着上面的说法：
test=Dog()
isinstance(test,Dog)
isinstance(test,Animal)

True
True
>>> 
说明子类从父类中继承了类型

这在传输参数的时候很有用，比如定义一个函数
def run_twice(animal):
    animal.run()
那么不管是上面的Dog也好，Cat也好，还是新定义的一个类
class Tortoise(Animal):
    def run(self):
        print('Tortoise is running slowly...')

由于都继承了Animal，所以无需对run_twice()进行修改


那么有个问题来了，Python作为动态语言
def run2(animal):
    animal.run()

class Animal(object):
    def run(self):
        print('Animal is running...')
class Dog(Animal):
    def run(self):
        print('Dog is running...')
class Timer(object):
    def run(self):
        print('Start...')

run2(Dog())
run2(Timer())


即使 Timer()类不是继承Animal但是，run2依旧可以运行，且仅仅需要
Timer()类里面有run 就行了，这是因为一些都是由 object 继承过来的！（吧

————————————————————————
获取对象信息

当我们拿到一个对象的引用时，如何知道这个对象是什么类型、有哪些方法呢？

type 从来给出XX是什么类型

>>> type(1253)
<class 'int'>
>>> type(abs)
<class 'builtin_function_or_method'>

那么继承的东东呢？
class Animal(object):
    def run(self):
        print('Animal is running...')
class Dog(Animal):
    def run(self):
        print('Dog is running...')

a=Dog()
print(type(a))

<class '__main__.Dog'>
>>> 
似乎没意思

————————————————————————
>>> len('ABC')
3
>>> 'ABC'.__len__()
3


class MyDog(object):
     def __len__(self):
         return 100

dog = MyDog()
dogg='1256'
print(len(dogg))
print(len(dog))
4
100
>>> 
优先输出。。诡异

————————————————————————
使用__slots__
class Student(object):
    __slots__ = ('name', 'age') # 用tuple定义允许绑定的属性名称
>>> s = Student() # 创建新的实例
>>> s.name = 'Michael' # 绑定属性'name'
>>> s.age = 25 # 绑定属性'age'
>>> s.score = 99 # 绑定属性'score'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Student' object has no attribute 'score'


使用__slots__要注意，__slots__定义的属性仅对当前类实例起作用，
对继承的子类是不起作用的：

>>> class GraduateStudent(Student):
...     pass
...
>>> g = GraduateStudent()
>>> g.score = 9999


————————————————————————
使用@property

Python内置的@property 
装饰器就是负责把一个方法变成属性调用的
（这部分我跳过了，没什么意义感觉）

————————————————————————
多重继承
某种东西既可以按A分类，也可以按B分类，如果统一搞一个继承就指数上升复杂噜
http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318680104044a55f4a9dbf8452caf71e8dc68b75a18000

class Dog(Mammal, Runnable):
    pass
可以进行这样的多重继承（有意思
这种设计通常称之为MixIn
为了更好地看出继承关系，Runnable → RunnableMixIn
class MyTCPServer(TCPServer, ForkingMixIn):
    pass

————————————————————————
定制类（ __str__ ）
class Student(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'Student object (name=%s)' % self.name
    __repr__ = __str__

-1.如果不使用 __str__ 那么 
print(Student('Michael'))
<__main__.Student object at 0x109afb190>

-2.如果使用 __str__ 那么
>>> print(Student('Michael'))
Student object (name: Michael)

-3.如果s=Student('Michael')，那么s直接敲还是会像1一样的，所以加上 __repr__ = __str__

————————————————————————
定制类（ __iter__ ）

class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1 # 初始化两个计数器a，b

    def __iter__(self):
        return self # 实例本身就是迭代对象，故返回自己

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b # 计算下一个值
        if self.a > 100000: # 退出循环的条件
            raise StopIteration()
        return self.a # 返回下一个值
for n in Fib():
    print(n)

1
1
2
3
...
46368
75025
>>> 

————————————————————————
定制类（ __getitem__）
由于上面的 __iter__ 不能像 list 一样使用，所可以考虑用 __getitem__ 类：
class Fib(object):
    def __getitem__(self, n):
        a, b = 1, 1
        for x in range(n):
            a, b = b, a + b
        return a
>>> f = Fib()
>>> f[0]
1


————————————————————————
定制类（ __getattr__）
不明觉厉
http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014319098638265527beb24f7840aa97de564ccc7f20f6000
————————————————————————

定制类（ __call__ ）
直接在实例本身上调用

class Student(object):
    def __init__(self, name):
        self.name = name

    def __call__(self):
        print('My name is %s.' % self.name)

>>> s = Student('Michael')
>>> s() # self参数不要传入
My name is Michael.

用 callable 判断对象是否有 __call__()的类实例
>>> callable(Student())
True
>>> callable(max)
True
>>> callable([1, 2, 3])
False

————————————————————————
枚举类
类似于C语言中的#define t 15
如果直接定义，如月份JAN=1，那么JAN依然是int 的变量，很危险
http://www.cnblogs.com/ucos/p/5896861.html

from enum import Enum, unique
@unique#如果有相同就会报错
class Color(Enum):
    red = 1
    yellow=2
    
red_member = Color.red
print(red_member.name)
print(red_member.value)#通过成员，来获取它的名称和值
print(type(Color.red))
for color in Color:#支持迭代
    print(color)

print(Color(1))#通过成员值来获取成员
print(Color['red'])#通过成员的名称来获取成员

red
1
<enum 'Color'>#不是int类
Color.red
Color.yellow#如果有别名那么就只会循环第一个出来

Color.red
Color.red
>>> 


特殊情况，
from enum import Enum
#用@unique就可以对red_alias报错
class Color(Enum):
    red = 1
    orange = 2
    yellow = 3
    green = 4
    blue = 5
    indigo = 6
    purple = 7
    red_alias = 1#默认情况下，不同的成员值允许相同。
    #但是两个相同值的成员，第二个成员的名称被视作第一个成员的别名

for color in Color.__members__.items():#要把别名也弄出来的话
    print(color)

('red', <Color.red: 1>)
('orange', <Color.orange: 2>)
('yellow', <Color.yellow: 3>)
('green', <Color.green: 4>)
('blue', <Color.blue: 5>)
('indigo', <Color.indigo: 6>)
('purple', <Color.purple: 7>)
('red_alias', <Color.red: 1>)
>>> 

————————————————————————
使用元类
我选择跳过。。似乎很少用到

————————————————————————
文件读写**
和C预言类似：
f = open('test.txt', 'r')
print(f.read())#输出txt文件中的东西
f.close()

有些时候f读入失败等情况会使得 close()失败
可以用with语句自动运行close
with open('test.txt', 'r') as f:
    print(f.read())

如果文件很大的话，read(size)比较保险，比如
print(f.read(1)) 只读一个字节（那怎么读下一个？循环

注意
with open('test.txt', 'r') as f:
    print(f.read(2).strip())
    print(f.read(3))
1
2
3
>>> 

with open('test.txt', 'r') as f:
    print(f.read(1).strip())
    print(f.read(2))
1

2
>>> 

with open('test.txt', 'r') as f:
    print(f.read(1))
    print(f.read(3))
1

2

>>> 
一方面，似乎read()和之前的next()效果类似，其次上面主要是'\n'作怪



一次读一行用readlines()
with open('test.txt', 'r') as f:
    for line in f.readlines():
        print(line.strip()) # 把末尾的'\n'删掉

————————————————————————
二进制文件
>>> f = open('/Users/michael/test.jpg', 'rb')
>>> f.read()
b'\xff\xd8\xff\xe1\x00\x18Exif\x00\x00...' # 十六进制表示的字节
对于视频和图片可以用 'rb' 来读取

如果是非UTF-8的文件的话
>>> f = open('/Users/michael/gbk.txt', 'r', encoding='gbk')
>>> f.read()
'测试'

有时候对于一些不规范的文件，可能会读取GG，最好的办法是忽略之
f = open('/Users/michael/gbk.txt', 'r', encoding='gbk', errors='ignore')

————————————————————————
写入文件

f = open('test.txt', 'w')
f.write('Hello, world!')
f.close()
如果没有写close的话就不一定完成写入，保险的做法是
with open('/Users/michael/test.txt', 'w') as f:
    f.write('Hello, world!')

用'a'是在下一行追加
f = open('test.txt', 'a')


————————————————————————
StringIO和BytesIO（str数据

StringIO → 内存中读写str

from io import StringIO
f=StringIO()
f.write('hello')
f.write('\n')
f.write('dedekinds')
print(f.getvalue())

hello
dedekinds
>>> 

from io import StringIO
f = StringIO('Hello!\nHi!\nGoodbye!')
while True:
     s = f.readline()
     if s == '':
         break
     print(s.strip())
Hello!
Hi!
Goodbye!
>>> 

————————————————————————
BytesIO（二进制数据

>>> from io import BytesIO
>>> f = BytesIO()
>>> f.write('中文'.encode('utf-8'))
6
>>> print(f.getvalue())
b'\xe4\xb8\xad\xe6\x96\x87'


>>> from io import BytesIO
>>> f = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
>>> f.read()
b'\xe4\xb8\xad\xe6\x96\x87'


————————————————————————
操作文件和目录

import os
pathtemp=os.path.abspath('.')#查看当前目录
#print(pathtemp)
pathtemp2=os.path.join(pathtemp, 'testdir')
# 在某个目录下创建一个新目录，首先把新目录的完整路径表示出来:
os.mkdir(pathtemp2)
# 然后创建一个目录:
os.rmdir(pathtemp2)
#删掉目录


拆分路径
>>>os.path.split('/Users/michael/testdir/file.txt')
('/Users/michael/testdir', 'file.txt')

>>> os.path.splitext('/path/to/file.txt')
('/path/to/file', '.txt')


这些合并、拆分路径的函数并不要求目录和文件要真实存在，
它们只对字符串进行操作。

————————————————————————
文件操作****
# 对文件重命名:
>>> os.rename('test.txt', 'abcc.txt ')#但是为什么改为'test.py'缺不行嗯？
os.rename('test.jpg', 'test.png')这个可以

# 删掉文件:
>>> os.remove('test.py')

import os
L=[x for x in os.listdir('.') if os.path.isdir(x)]
print(L)
查看当前目录的文件夹名字

import os
L=[x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.py']
print(L)
获取当前所有py文件，似乎MATLAB更加方便


————————————————————————
正则表达式（感觉需要找个项目来练练

'\d'数字 '\w'字母 '.'任意字符
-用*表示任意个字符（包括0个）
-用+表示至少一个字符（？？？）
-用?表示0个或1个字符
-用{n}表示n个字符
-用{n,m}表示n-m个字符
-'\s'表示一个空格

'\d{3}\s+\d{3,8}'表示任意个空格隔开的带区号的电话号码
'\d{3}\-\d{3,8}'匹配'010-12345'

-A|B可以匹配A或B，所以(P|p)ython可以匹配'Python'或者'python'。
-^表示行的开头，'^\d'表示必须以数字开头。
-$表示行的结束，'\d$'表示必须以数字结束。

import re
t=re.match(r'^\d{3}\-\d{3,8}$', '010-12345')
print(t)

<_sre.SRE_Match object; span=(0, 9), match='010-12345'>
>>> 

python强烈建议用下方的r前缀表达字符串，不用考虑转义
s = r'ABC\-001' # Python的字符串
# 对应的正则表达式字符串不变：
# 'ABC\-001'
————————————————————————
切分字符串（？？？

>>> 'a b   c'.split(' ')
['a', 'b', '', '', 'c']

>>> re.split(r'\s+', 'a b     c')
['a', 'b', 'c']

>>> re.split(r'[\s\,\;]+', 'a,b;; c  d')
['a', 'b', 'c', 'd']

import re
m=re.match(r'^(\d{3})-(\d{3,8})$','010-21198')#加（）表示要提取的部分
print(m.group(0))#原字符
print(m.group(1))
print(m.group(2))

010-21198
010
21198
>>> 

————————————————————————
贪婪匹配
#？？？？
>>> re.match(r'^(\d+)(0*)$', '102300').groups()
('102300', '')
#？？？？
>>> re.match(r'^(\d+?)(0*)$', '102300').groups()
('1023', '00')


作业：
1.请尝试写一个验证Email地址的正则表达式。版本一应该可以验证出类似的Email：
someone@gmail.com
bill.gates@microsoft.com

2.版本二可以验证并提取出带名字的Email地址：
<Tom Paris> tom@voyager.org


————————————————————————
常用内建模块***
1.datetime 处理日期和时间的标准库库

from datetime import datetime
now=datetime.now()
print(now)
2017-05-14 16:52:43.882699
>>> 

或者

import datetime
now=datetime.datetime.now()
print(now)
2017-05-14 16:56:55.931115
>>> 


创建一个时间
>>> from datetime import datetime
>>> dt = datetime(2015, 4, 19, 12, 20) # 用指定日期时间创建datetime
>>> print(dt)
2015-04-19 12:20:00

-datetime转换为timestamp
timestamp = 0 = 1970-1-1 08:00:00 UTC+8:00 北京时间的时间元年表达(实际上和时区无关)

from datetime import datetime
dt=datetime(2017,5,20,12,30)
print(dt.timestamp())

1495254600.0
>>> 


-timestamp转换为datetime
>>> from datetime import datetime
>>> t = 1429417200.0
>>> print(datetime.fromtimestamp(t))
2015-04-19 12:20:00

>>> print(datetime.utcfromtimestamp(t)) # UTC时间，相差8小时
2015-04-19 04:20:00


str转换为datetime
>>> from datetime import datetime
>>> cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
>>> print(cday)
2015-06-01 18:19:59

还有各种关于时间的操作
http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431937554888869fb52b812243dda6103214cd61d0c2000

————————————————————————
collections

from collections import namedtuple#定义一个坐标
Point=namedtuple('Point',['x','y'])
p=Point(1,2)
print(p.x,p.y)
1 2
>>> 

如果要用坐标和半径表示一个圆
# namedtuple('名称', [属性list]):
Circle = namedtuple('Circle', ['x', 'y', 'r'])

————————————————————————
deque
list 的append()和pop()功能可以更好地处理，由于是线性存储
插入和删除效率不高?，可以用下面的办法

from collections import deque
q=deque(['a','b','c'])
q.append('x')
q.appendleft('y')#deque是双向列表，适合用于队列和栈

还有pop(),popleft()

————————————————————————
Counter#实际上是dict的一个子类
下面这个例子就是统计又给单词中出现字母的频率

from collections import Counter
c=Counter()
for ch in 'aabbbccccc':
    c[ch]=c[ch]+1
print(c)

Counter({'c': 5, 'b': 3, 'a': 2})
>>> 

————————————————————————
hashlib

import hashlib
md5=hashlib.md5()
md5.update('how to use md5 in'.encode('utf-8'))#太长的话可以分段
md5.update('python hashlib??'.encode('utf-8'))
print(md5.hexdigest())

b5e2e2cc16973f6a35420e8b29ffe266
>>> 

-即使改动一个字母变化都会很大（为什么？
可以换成sha1=hashlib.sha1()

-运营商存储密码明文是不科学的，应该存储hashlib后的码，即使泄漏也不可逆向

-黑客有可能事先对简单的密码处理了md5，所以垃圾密码很容易被盗号
但是运营商应该可以对用户的简单密码稍微添加一点字符，再md5即可


————————————————————————
itertools 迭代工具

1.自然数序列无限：
import itertools
natuals=itertools.count(1)
for n in natuals:
    print(n)

1
2
3
4
5
>>> 

2.字符串无限重复：
>>> import itertools
>>> cs = itertools.cycle('ABC') # 注意字符串也是序列的一种
>>> for c in cs:
...     print(c)
...
'A'
'B'
'C'
'A'


3.元素无限重复：
>>> ns = itertools.repeat('A', 3)#第二个元素表示重复次数
>>> for n in ns:
...     print(n)
...
A
A
A


除了用for做一个判断之外，还可以用takewhile()来截取无限数列
>>> natuals = itertools.count(1)
>>> ns = itertools.takewhile(lambda x: x <= 10, natuals)
>>> list(ns)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

特别地

1.chain()可以将迭代对象组合起来，形成更大的迭代器：
>>> for c in itertools.chain('ABC', 'XYZ'):
...     print(c)
# 迭代效果：'A' 'B' 'C' 'X' 'Y' 'Z'

2.groupby()把迭代器中相邻的重复元素挑出来放在一起：
>>> for key, group in itertools.groupby('AAABBBCCAAA'):
...     print(key, list(group))
...
A ['A', 'A', 'A']
B ['B', 'B', 'B']
C ['C', 'C']
A ['A', 'A', 'A']

————————————————————————
contextlib（简化上下文管理）

用with语句可以让f open 后及时被 close ，
with open('/path/to/file', 'r') as f:
    f.read()

类似地也可以这样用with语句
class Query(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print('Begin')
        return self#必须返回，没有的话query没有输入

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print('Error')#__exit__的参数不懂？
        else:
            print('End')

    def query(self):
        print('Query info about %s...' % self.name)

with Query('Bob') as q:
    q.query()

Begin
Query info about Bob...
End
>>> 

上面编写__enter__和__exit__仍然很繁琐

from contextlib import contextmanager

class Query(object):

    def __init__(self, name):
        self.name = name

    def query(self):
        print('Query info about %s...' % self.name)

@contextmanager
def create_query(name):
    print('Begin')
    q = Query(name)
    yield q
    print('End')

with create_query('Bob') as q:
    q.query()

Begin
Query info about Bob...
End
>>> 

还能这么玩
from contextlib import contextmanager
@contextmanager
def tag(name):
    print("<%s>" % name)
    yield
    print("</%s>" % name)

with tag("h1"):
    print("hello")
    print("world")

<h1>
hello
world
</h1>
>>> 



1.with语句首先执行yield之前的语句，因此打印出<h1>；
2.yield调用会执行with语句内部的所有语句，因此打印出hello和world；
3.最后执行yield之后的语句，打印出</h1>。

————————————————————————
urllib**
