import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

def create_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape))


def create_model(x):
    y_predict=0
    with tf.variable_scope("conv1"):
        #修改型状
        input_x=tf.reshape(x,shape=[-1,28,28,1])
        conv1_weights=create_weights(shape=[5,5,1,32])
        conv1_bias=create_weights(shape=[32])
        conv1_x=tf.nn.conv2d(input_x,filter=conv1_weights,strides=[1,1,1,1],padding='SAME')+conv1_bias
        relu_x=tf.nn.relu(conv1_x)
        pool1_x=tf.nn.max_pool(relu_x,[1,2,2,1],[1,2,2,1],padding="SAME")
    with tf.variable_scope("conv2"):
        conv2_weights = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_bias
        relu2_x = tf.nn.relu(conv2_x)
        pool2_x = tf.nn.max_pool(relu2_x, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    with tf.variable_scope("fullconnection"):
        X_FC=tf.reshape(pool2_x,shape=[-1,7*7*64])
        weights_fc=create_weights(shape=[7*7*64,10])
        bias_fc=create_weights(shape=[10])
        y_predict=tf.matmul(X_FC,weights_fc)+bias_fc
    return y_predict


def full_connection():
#准备数据
    mnist=input_data.read_data_sets("G:\CNN\mnist",one_hot=True)
    with tf.variable_scope("mnist_data"):
        x=tf.placeholder(dtype=tf.float32,shape=(None,784))
        y_true=tf.placeholder(dtype=tf.float32,shape=(None,10))
#构建模型
    y_predict=create_model(x)
#构建损失函数
    with tf.variable_scope("loss"):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
#优化损失
    with tf.variable_scope("optimizer"):
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
#准确率计算
    equal_list=tf.equal(tf.argmax(y_true,1),
                        tf.argmax(y_predict,1))
    accuracy=tf.reduce_mean(tf.cast(equal_list,tf.float32))
    tf.summary.scalar("loss",loss)
    tf.summary.scalar("Acc",accuracy)
    init=tf.global_variables_initializer()
    merge=tf.summary.merge_all()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        filewriter=tf.summary.FileWriter('G:/CNN/tmp',graph=sess.graph)
        image,label=mnist.train.next_batch(100)
        for i in range(1000):
            _,error,accuracy_value=sess.run([optimizer,loss,accuracy],feed_dict={x:image,y_true:label})
            print('第%d次的损失为%f,准确率为%f'%(i+1,error,accuracy_value))
            summary=sess.run(merge,feed_dict={loss:error,accuracy:accuracy_value})
            filewriter.add_summary(summary,i)
            if i%100==0:
                saver.save(sess,'G:/CNN/tmp')
    return None



if __name__=="__main__":
    full_connection()














