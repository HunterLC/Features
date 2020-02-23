import tensorflow as tf
import numpy as np

#数据集准备
batch_size =100; w =10; h =50; c =1
X = np.random.uniform(0,1,(batch_size, w, h, c))
Y = np.random.randint(0,2,(batch_size))
print(X)
print(Y)

#图模型--占位符
G_X = tf.placeholder(tf.float32, [None, w, h, c])
G_Y = tf.placeholder(tf.int32, [None])

#权重矩阵初始化
def weight_variable(shape):
    #用正太分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏量初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积层定义

def con2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#池化层定义

def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#cnn--卷积层 过滤器形状5x2x1设置8个
W_conv1 = weight_variable([5,2,1,8])
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(con2d(G_X, W_conv1) + b_conv1)

#cnn--池化层
h_pool1 = max_pool_2d(h_conv1)

#flatten层  扁平化处理
f_flatten = tf.reshape(h_conv1, [-1,5*25*8])

#全连接,处理成类似标签的样子
W_f1 = weight_variable([5*25*8, 1])
b_f1 = bias_variable([1])
h_res = tf.nn.sigmoid(tf.matmul(f_flatten, W_f1) + b_f1)
res = tf.reshape(h_res, [-1])

with tf.Session()as sees:
    sees.run(tf.global_variables_initializer())
    # print(sees.run(h_conv1, feed_dict={G_X: X, G_Y: Y}))
    # print(h_conv1)
    # print(type(h_conv1))
    # print(sees.run(h_pool1, feed_dict={G_X: X, G_Y: Y}))
    # print(h_pool1)
    # print(type(h_pool1))
    # print(sees.run(res, feed_dict={G_X: X, G_Y: Y}))
    # print(res)
    # print(type(res))
    # print(G_Y)
