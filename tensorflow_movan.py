
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
    #注意x的维度，是行向量
    Weights = tf.Variable(tf.random_normal([in_size ,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs ,Weights) + biases
    
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output




x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
    #[:, np.newaxis]在这里起到的转置的作用
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs=tf.placeholder(tf.float32,[None,1])
    #tf.float32不能省略
    #[None,1]中的None表示可以是任意的数值
ys=tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs ,1 ,10 ,tf.nn.relu)
prediction = add_layer(l1 ,10 ,1 ,activation_function=None)

loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys-prediction) ,reduction_indices=[1] ) )
    #reduction_indices=[1]实际上就是MATLAB中sum(a,1) 行or sum(a,2)列的作用
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(train_step ,feed_dict={xs:x_data ,ys:y_data})
        if i%50==0:
            print(sess.run(loss ,feed_dict={xs:x_data ,ys:y_data}))
        


_____________________________________________________________________________________
训练可视化
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

‘’‘
同上
’‘’

with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)#*subplot(a,b,c),将画面变为a*b，然后放置在第c个
    ax.scatter(x_data,y_data)
    plt.ion()#打开交互模式，show()之后程序不暂停，类似hold on ?
    plt.show()
    for i in range(10000):
        sess.run(train_step ,feed_dict={xs:x_data ,ys:y_data})
        if i%50==0:
            #print(sess.run(loss ,feed_dict={xs:x_data ,ys:y_data}))
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            ax.lines.remove(lines[0])
            plt.pause(0.1)#需要在pyplot.py上加入import time

#没能成功在原有的scatter上生成所需要的线段==、
        
        

#——————————————————————————————————————————
利用MINST数据进行classfication的练习

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#one_hot限制格式为0 1

def add_layer(inputs ,in_size ,out_size ,activation_function=None):
    #注意x的维度，是行向量
    Weights = tf.Variable(tf.random_normal([in_size ,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs ,Weights) + biases
    
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        #y_pre is a real vector between [0,1]
        #tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False
        #tf.argmax(vector, 1)：返回的是vector中的最大值的索引号
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.cast(x, dtype, name=None)
        #x:input
        #dtype:target type
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
    
    
xs=tf.placeholder(tf.float32,[None,784])#28*28=784
ys=tf.placeholder(tf.float32,[None,10])#

prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)#取下一波100组数据
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
                print(compute_accuracy(mnist.test.images, mnist.test.labels))#计算测试集的准确性
                #print(sess.run(cross_entropy,feed_dict={xs: batch_xs, ys: batch_ys}))#输出误差（交叉熵

                
#——————————————————————————————————————————————————————————————

dropout的话，keep_prob的数值对结果有较大的影响
keep_prob
        
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        if i%100==0:
            print(sess.run(cross_entropy,feed_dict={xs: X_test, ys: y_test, keep_prob: 1}))
            #*在测试的时候应该保持keep_prob为1，成为一个完整的网络


#——————————————————————————————————————————————————————————
CNN卷积神经网路

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,mean=0.0,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#*[1,x_movemont,y_movement,1,]
    #'VALID':add notining
    #'SAME':add 0

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    '''
    (1)value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape 
    (2)ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1  
    (3)strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    (4)padding：和卷积类似，可以取'VALID' 或者'SAME'
    (5)返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    '''


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])#/255.? # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])#*-1代表先不考虑输入的图片例子多少这个维度

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32])#patch 5*5   out:1channels depth=32?
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#28*28*32
h_pool1 = max_pool_2x2(h_conv1)#14*14*32



## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#14*14*64
h_pool2 = max_pool_2x2(h_conv2)##7*7*64

## func1 layer ##
W_fc1 = weight_variable([7*7*64,1024]) 
b_fc1 = bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64]) #flat [n_samples,7,7,64]-->[n_sample,7*7*64]

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


## func2 layer ##
W_fc2 = weight_variable([1024,10]) 
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)




# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#AdamOptimizer

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))



        
        
