
_____________________________________________
单输入+单神经元
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:21:41 2018

"""

import tensorflow as tf
import numpy as np
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.142857 + 0.3



#NN structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))#tf.random_uniform([one_dim],a,b)
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)


#training

init = tf.global_variables_initializer()  # init the arg

sess = tf.Session()
sess.run(init)          # Very important

for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases),sess.run(loss))


_____________________________________________________________________________________
Session
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:21:41 2018
"""

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1 ,matrix2)


###method 1
#sess = tf.Session()
#result = sess.run(product)
#print(result)
##sess.close

###method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

_____________________________________________________________________________________
Variable
new_value  update看成是函数
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:21:41 2018
"""

import tensorflow as tf
import numpy as np

state = tf.Variable(0,name='name')
#print(state.name)

# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)

# 如果定义 Variable, 就一定要 initialize
# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好
# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
_____________________________________________________________________________________

#这一次我们会讲到 Tensorflow 中的 placeholder , placeholder 是 Tensorflow 中的占位符，暂时储存变量.
#Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 
#然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).


import tensorflow as tf

#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32,[2,2])
input2 = tf.placeholder(tf.float32,[2,2])

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [[7,2],[2,1]], input2: [[2,3],[3,6]]}))


# [[ 14.   6.]
# [  6.   6.]]


_____________________________________________________________________________________

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def add_layer(inputs ,in_size ,out_size ,activation_function=None):
    
    Weights = tf.Variable(tf.random_normal([in_size ,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs ,Weights) + biases
    
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output




x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs ,1 ,10 ,tf.nn.relu)
prediction = add_layer(l1 ,10 ,1 ,activation_function=None)

loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys-prediction) ,reduction_indices=[1] ) )
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(train_step ,feed_dict={xs:x_data ,ys:y_data})
        if i%50==0:
            print(sess.run(loss ,feed_dict={xs:x_data ,ys:y_data}))
        