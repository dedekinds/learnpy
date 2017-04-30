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
for key in d():
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
y = B
x = A
z = C

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


杨辉三角
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
3.集合数据类型如list、dict、str等是Iterable但不是Iterator，不过可以通过iter()函数获得一个Iterator对象。
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
    s=[f(x) for f in fs]
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
    return lambda x: x % n > 0


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
返回函数
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
def bar():
    print('i am bar')

bar=use_logging(bar)
bar()

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
    #这句话的意思就是，当模块被直接运行时，
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



