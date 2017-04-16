#1
print('hello %s %s %%',%('abx','sss'))#和C语言类似的结构化
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
