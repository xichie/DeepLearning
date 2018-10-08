import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 
import numpy as np
from scipy import misc,ndimage
import tensorflow.contrib.slim as slim
mnist = input_data.read_data_sets("./MNIST_data")
batch_size = 100
width,height = 28,28
tf.reset_default_graph()
#生成器
def G(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0 #判断reuse是否是true
    # for t in tf.global_variables():    上一句话相当于下面这三行
    #     if t.name.startswith('generator'):
    #         reuse = len (t) > 0
    with tf.variable_scope('generator',reuse = reuse):
        x = slim.fully_connected(x,32,activation_fn = tf.nn.relu)
        x = slim.fully_connected(x,128,activation_fn = tf.nn.relu)
        x = slim.fully_connected(x,28*28,activation_fn = tf.nn.sigmoid)
        return x
#判别器
def D(x):#python 的推断表达
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    # for t in tf.global_variables():    #上一句话相当于下面这三行
    #     if t.name.startswith('discriminator'):
    #         reuse = len (t) > 0
    with tf.variable_scope('discriminator',reuse = reuse):
        x = slim.fully_connected(x,128,activation_fn = tf.nn.relu)
        x = slim.fully_connected(x,32,activation_fn = tf.nn.relu)
        x = slim.fully_connected(x,1,activation_fn = tf.nn.sigmoid)
    return x


real_X = tf.placeholder(tf.float32,shape = [batch_size,28*28])
random_z = tf.placeholder(tf.float32,shape = [batch_size,100])
generator_X = G(random_z)
# tf.set_random_seed = 1
epslion = tf.random_uniform([batch_size,1],minval = 0,maxval = 1)
inter_X = real_X * epslion + generator_X * (1-epslion)
def pen(X):
	inter_X = tf.gradients(D(X),X)[0]
	inter_X_norm = tf.sqrt(tf.reduce_sum(inter_X**2))
	pen_X = 10*tf.reduce_mean(tf.nn.relu(inter_X_norm-1))#lipschiz设为1
	return pen_X
D_loss = -tf.reduce_mean(tf.log(D(real_X))+ tf.log(1 - D(generator_X))) + pen(inter_X)
G_loss = -tf.reduce_mean(tf.log(D(generator_X)))
t_vars = tf.trainable_variables()
# for var in t_vars:
# 	if 'discriminator' in var.name:
# 		d_vars = var
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]
#print (len(t_vars),len(d_vars),len(g_vars))
D_opt = tf.train.RMSPropOptimizer(1e-2).minimize(D_loss,var_list = d_vars)
G_opt = tf.train.RMSPropOptimizer(1e-2).minimize(G_loss,var_list = g_vars)
epochs = 100

with tf.Session() as sess:
	print("start。。。。")
	sess.run(tf.global_variables_initializer())
	if not os.path.exists('out/'):
		os.makedirs('out/')
	for epoch in range(epochs):
		total_batch = int(mnist.train.num_examples/batch_size)
		for e in range(total_batch):
			for i in range(5):#判别器训练5次，生成器训练一次
				real_batch_X,real_batch_Y = mnist.train.next_batch(batch_size)
				random_batch_Z = sess.run(tf.random_uniform([batch_size,100],minval=-1,maxval=1))
				D_loss_,_= sess.run([D_loss,D_opt],feed_dict = {real_X:real_batch_X,random_z:random_batch_Z})
			#固定判别器，训练生成器
			random_batch_Z = sess.run(tf.random_uniform([batch_size,100],minval=-1,maxval=1))
			G_loss_,_ = sess.run([G_loss,G_opt],feed_dict = {random_z:random_batch_Z})
			#可视化
			if e % 50 == 0:
				print('e %s,D_loss:%s,G_loss:%s'%(e,D_loss_,G_loss_))
				n_rows = 6
				check_imgs = sess.run(generator_X,{random_z:random_batch_Z}).reshape((batch_size,width,height))[:n_rows*n_rows]   
				imgs = np.ones((width*n_rows+5*n_rows+5,width*n_rows+5*n_rows+5))
				for i in range(n_rows*n_rows):
					num1 = (i%n_rows)
					num2 = np.int32(i/n_rows)
					imgs[5+5*num1+width*num1:5+5*num1+width+width*num1,5+5*num2+height*num2:5+5*num2+height+height*num2] = check_imgs[i]
					#图片与图片之间有留白，大小为5，四个点确定一个图片的位置，如上
				misc.imsave('out/%s.png'%(e/100), imgs)
	print('完成！')