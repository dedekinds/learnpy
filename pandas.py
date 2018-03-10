#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 21:45:20 2018

"""

import pandas as pd
import numpy as np
s = pd.Series([1,3,np.nan,5])
dates = pd.date_range('20160101',periods=6)

#__________________________________________________________
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
'''
                   a         b         c         d
2016-01-01  0.016718  0.942491 -0.269798  0.355269
2016-01-02 -0.531919 -0.094367 -0.884143 -0.548323
2016-01-03  2.225073 -0.285678 -0.159767 -0.026743
2016-01-04  0.192052 -0.440277 -0.997777  0.200826
2016-01-05 -0.454307  0.798392 -1.193261  0.148836
2016-01-06  1.812445 -0.292142  1.907257 -1.535865
'''
#__________________________________________________________
df1=pd.DataFrame(np.arange(12).reshape((3,4)))
'''
   0  1   2   3
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
'''
#__________________________________________________________
df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
                    
print(df2)
'''
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
'''
#__________________________________________________________

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
print(df['A'],df.A)
'''
2013-01-01     0
2013-01-02     4
2013-01-03     8
2013-01-04    12
2013-01-05    16
2013-01-06    20
'''

print(df[0:3])#取数据

'''
            A  B   C   D
2013-01-01  0  1   2   3
2013-01-02  4  5   6   7
2013-01-03  8  9  10  11
'''

print(df.loc['20130103'])
'''
A     8
B     9
C    10
D    11
'''

print(df.loc['20130103',['A','B']])
'''
A    8
B    9
Name: 2013-01-03 00:00:00, dtype: int64
'''
print(df.iloc[3,1])
'''
13
'''

df.iloc[2,2]=23333#修改数据
df.loc['20130105','B']=23333

'''
             A      B      C   D
2013-01-01   0      1      2   3
2013-01-02   4      5      6   7
2013-01-03   8      9  23333  11
2013-01-04  12     13     14  15
2013-01-05  16  23333     18  19
2013-01-06  20     21     22  23
'''

df.A[df.A>4]='test'
'''
               A   B   C   D
2013-01-01     0   1   2   3
2013-01-02     4   5   6   7
2013-01-03  test   9  10  11
2013-01-04  test  13  14  15
2013-01-05  test  17  18  19
2013-01-06  test  21  22  23
'''

df['F']=np.nan#增加一行数据
'''
               A   B   C   D   F
2013-01-01     0   1   2   3 NaN
2013-01-02     4   5   6   7 NaN
2013-01-03  test   9  10  11 NaN
2013-01-04  test  13  14  15 NaN
2013-01-05  test  17  18  19 NaN
2013-01-06  test  21  22  23 NaN
'''
df['E']=pd.Series([1,2,3,4,5,6],index=pd.date_range('20130101',periods=6))
'''
               A   B   C   D   F  E
2013-01-01     0   1   2   3 NaN  1
2013-01-02     4   5   6   7 NaN  2
2013-01-03  test   9  10  11 NaN  3
2013-01-04  test  13  14  15 NaN  4
2013-01-05  test  17  18  19 NaN  5
2013-01-06  test  21  22  23 NaN  6
'''
print(df.dropna(axis=1,how='any'))#axis=0,how='all'
'''
               A   B   C   D  E
2013-01-01     0   1   2   3  1
2013-01-02     4   5   6   7  2
2013-01-03  test   9  10  11  3
2013-01-04  test  13  14  15  4
2013-01-05  test  17  18  19  5
2013-01-06  test  21  22  23  6
'''
print(df.fillna(value=0))
'''
               A   B   C   D    F  E
2013-01-01     0   1   2   3  0.0  1
2013-01-02     4   5   6   7  0.0  2
2013-01-03  test   9  10  11  0.0  3
2013-01-04  test  13  14  15  0.0  4
2013-01-05  test  17  18  19  0.0  5
2013-01-06  test  21  22  23  0.0  6
'''
print(df.isnull())
'''
                A      B      C      D     F      E
2013-01-01  False  False  False  False  True  False
2013-01-02  False  False  False  False  True  False
2013-01-03  False  False  False  False  True  False
2013-01-04  False  False  False  False  True  False
2013-01-05  False  False  False  False  True  False
2013-01-06  False  False  False  False  True  False
'''

import pandas as pd

data = pd.read_csv('/home/dedekinds/student.csv')

data.to_pickle('student.pickle')


#####
import pandas as pd
import numpy as np
#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

#concat纵向合并
res = pd.concat([df1, df2, df3], axis=0,ignore_index=True)



######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#series
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()


#dataframe
data = pd.DataFrame(np.random.randn(1000,4),
                    index = np.arange(1000),
                    columns = list("ABCD"))

data = data.cumsum()#累加
data.plot()


####
ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')#散点图
# 将之下这个 data 画在上一个 ax 上面
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()




