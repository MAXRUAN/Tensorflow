import tensorflow as tf
import load_cifar10
import  numpy as np
batch_size=100
train_iters=3000
learning_rate=0.01
display_step=10
step=0

x_train=tf.placeholder(dtype=tf.float32,shape=[None,32,32,3])
y_train=tf.placeholder(dtype=tf.float32,shape=[None,10])
keep_prob=tf.placeholder(dtype=tf.float32)
is_training=tf.placeholder(dtype=tf.bool)

#conv1
W1=tf.Variable(tf.truncated_normal\
              ([3,3,3,64],dtype=tf.float32,stddev=5e-2))
conv_1=tf.nn.conv2d(x_train,W1,strides=[1,1,1,1],padding="SAME")
bn1=tf.layers.batch_normalization(conv_1,training=is_training)
relu_1=tf.nn.relu(bn1)
print (relu_1)
pooling_1=tf.nn.max_pool(relu_1,\
           ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
print (pooling_1)

# conv2
W2=tf.Variable(tf.truncated_normal\
               ([3,3,64,128],dtype=tf.float32,stddev=5e-2))
conv_2=tf.nn.conv2d(pooling_1,W2,strides=[1,1,1,1],padding="SAME")
bn2=tf.layers.batch_normalization(conv_2,training=is_training)
relu_2=tf.nn.relu(bn2)
pooling_2=tf.nn.max_pool(relu_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
print  pooling_2

#conv3
W3=tf.Variable(tf.truncated_normal(\
              [3,3,128,256],dtype=tf.float32,stddev=5e-2))
conv_3=tf.nn.conv2d(pooling_2,W3,strides=[1,1,1,1],padding="SAME")
bn3=tf.layers.batch_normalization(conv_3,training=is_training)
relu_3=tf.nn.relu(bn3)
pooling_3=tf.nn.max_pool(relu_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print  pooling_3
#fc1

dense_tmp=tf.reshape(pooling_3,shape=[-1,4*4*256])
print (dense_tmp)

fc1=tf.Variable(tf.truncated_normal([4*4*256,1024],stddev=0.04))
bn_fc1=tf.layers.batch_normalization(tf.matmul(dense_tmp,fc1),training=is_training)
dense1=tf.nn.relu(bn_fc1)
dropout_1=tf.nn.dropout(dense1,keep_prob)
print  dropout_1
#fc2
fc2=tf.Variable(tf.truncated_normal(shape=[1024,10],dtype=tf.float32,stddev=0.04))
out=tf.matmul(dropout_1,fc2)
print "out", out

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y_train))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

dr=load_cifar10.Cifar10DataReader(cifar_folder="/home/wj/PycharmProjects/data-set/cifar-10/cifar-10-batches-py")

# test nets
correct_predict=tf.equal(tf.arg_max(out,1),tf.arg_max(y_train,1))
accuracy=tf.reduce_mean(tf.cast(correct_predict,tf.float32))

#init = tf.initialize_all_variables()
init=tf.global_variables_initializer()
saver=tf.train.Saver()

# start to train
with tf.Session() as sess:
    sess.run(init)
    step=0
    while step < train_iters:
        batch_xs,batch_ys=dr.next_train_data(batch_size)
        opt,acc,loss=sess.run([optimizer, accuracy, cost], feed_dict={x_train:batch_xs, y_train: batch_ys, keep_prob: 0.6, is_training: True})

        step+=1
        if step % display_step ==0 :
            print ("Iter "+ str(step*batch_size) +
                   ",MniBatch loss " + "{: .6f})".format(loss)+
                   ",training Acc " +"{: .6f}".format(acc))

# test the model
    num_exampels=10000
    d,l=dr.next_test_data(num_exampels)
    test_acc=sess.run(accuracy,feed_dict={x_train:d,y_train:l,keep_prob:1.0,is_training:True})
    print ("test ACC : ",test_acc)
    saver.save(sess,"model_temp/cifar10_cnn_demo.ckpt")













