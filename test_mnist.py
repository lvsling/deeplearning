#coding:utf-8
#引入mnist数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#定义一个save_pic()函数用来保存MNIST数据集图片
def save_pic():
	import scipy.misc
	import os
	#保存原始图片
	save_dir='MNIST_data/raw/'
	if os.path.exists(save_dir) is False:
		os.makedirs(save_dir)
	#保存前20张图片
	for i in range(20):
		image_array = mnist.train.images[i,:]
		image_array = image_array.reshape(28,28)
		filename = save_dir +'mnist_train_%d.jpg' % i
		scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)

	print('please check: %s' % save_dir)

#定义一个label()函数用来打印MNIST数据集中图片的数字。
def label():
	import numpy as np 
	#查看前20张训练图的lable
	for i in range(20):
		one_hot_label = mnist.train.labels[i,:]
		#np.argmax 可以直接获取原始的label
		label = np.argmax(one_hot_label)
		print('mnist_train_%d.jpg label:%d' % (i,label))

#softmax回归
def softmax_regression():
	import tensorflow as tf 
	#创建一个占位符x，表示待识别的图片
	x = tf.placeholder(tf.float32,[None,784])
	#定义weights
	w = tf.Variable(tf.zeros([784,10]))
	#定义biases
	b = tf.Variable(tf.zeros([10]))
	#y表示模型输出，--> y=softmax(wx+b)
	y=tf.nn.softmax(tf.matmul(x,w)+b)
	#y_表示的实际的图像数字标签
	y_ = tf.placeholder(tf.float32,[None,10])
	#根据y和y_构建交叉熵损失
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
	#用随机梯度下降针对模型的参数进行优化
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	#
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	print('start training...')
	#进行1000步梯度下降
	for i in range(1000):
		batch_xs,batch_ys = mnist.train.next_batch(100)
		sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

	#正确的预测结果
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	#计算预测准确率
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	#获取最终模型的正确率
	print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

#两层卷积网络分类

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.01)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def convolutional():

	#x为训练图像的占位符，y_为训练图像标签的占位符
	x = tf.placeholder(tf.float32,[None,784])
	y_ = tf.placeholder(tf.float32,[None,10])
	#将单张图片从784维向量重新还原为28*28的矩阵图片
	x_image = tf.reshape(x,[-1,28,28,1])
	#第一层卷积层
	W_conv1 = weight_variable([5,5,1,32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
	h_pool1 =max_pool_2x2(h_conv1)
	#第二层卷积层
	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	#全连接层
	W_fc1 = weight_variable([7 * 7 * 64,1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
	#使用Dropout，keep_prob是一个占位符，训练时为0.5 测试时为1
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
	#把1024维的向量转换为10维，对应10个类别
	W_fc2 = weight_variable([1024,10])
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2

	#采用tf.nn.softmax_cross_entropy_with_logits直接计算
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
	#定义train_step
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	#定义测试的准确率
	correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	#创建Session和变量初始化
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	#训练
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		#每100步报告一次在验证集上的准确度
		if i % 100 ==0:
			train_accuracy=accuracy.eval(feed_dict = {
				x: batch[0],y_: batch[1],keep_prob: 1.0
				})
			print('step %d, training accuracy %g' % (i,train_accuracy))
		train_step.run(feed_dict={x: batch[0],y_: batch[1],keep_prob:0.5})
	#训练结束后报告在测试集上准确显示
	print('test accuracy %g ' % accuracy.eval(feed_dict={
		x:mnist.test.images,y_: mnist.test.labels,keep_prob:1.0
		}))


	








if __name__ == '__main__':
	#调用保存MNIST数据集函数
	#save_pic()
	#打印图片上的数字
	#label()
	#softmax回归算法
	#softmax_regression()
	#两层卷积神经网络
	convolutional()


