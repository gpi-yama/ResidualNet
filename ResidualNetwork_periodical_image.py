# 
#
# (C) 2018 R.Yamaguchi
# Residual network
# non pooling
# image size = 64*128
#

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import multiprocessing as mp
import sys
import math

class ResCNNPredNet():
    def __init__(self, height, width, channels, config):
        self.config = config
        self.height = height
        self.channels = channels
        self.width = width
        self._x = None
        self._y = None
        self._sess = None
        self._n_batch = None
        self.is_training = None

    def weight_variable(self, shape=None, name=None):
        with tf.name_scope('weight_'+name):
            w = tf.Variable(tf.random_normal(shape=shape, stddev=0.01), name=name)
        return w
        
    def bias_variable(self, shape, name):
        with tf.name_scope('bias_'+name):
            b = tf.Variable(tf.zeros(shape),name=name)
        return b

    def inference(self,x):
        x_pad = tf.concat((x[:, :, :, self.width-3:], 
                           tf.concat((x, x[:, :, :, 0:3]), axis=3)), axis=3)
        x_pad = tf.pad(x_pad, [[0,0],[0,0],[3,3],[0,0]])
        inp = tf.nn.conv2d(x_pad, self.weight_variable(shape=[7, 7, 1, 32], name='first'),
                           strides=[1,1,1,1], padding='VALID', data_format='NCHW')
        #inp = tf.nn.relu(tf.nn.bias_add(inp, self.bias_variable([32], name='cb_inp'), data_format='NCHW'))
        inp = tf.nn.relu(inp)
        input_layer = self.ResidualBlock(inp, 32, 7, name='inp')
        
        layer2 = self.ResidualBlock(input_layer, 32, 5, name='2')
        layer2 = tf.nn.conv2d(layer2, self.weight_variable(
            shape=[1, 1, 32, 64], name='conv_layer2'),
                              strides=[1,1,1,1], padding='SAME', data_format='NCHW')

        layer3 = self.ResidualBlock(layer2, 64, 5, name='3')
        layer3 = tf.nn.conv2d(layer3, self.weight_variable(
            shape=[1, 1, 64, 32], name='conv_layer3'),
                              strides=[1,1,1,1], padding='SAME', data_format='NCHW')
        layer4 = self.ResidualBlock(layer3, 32, 5, name='4')

        layer5 = self.ResidualBlock(layer4, 32, 7, name='5')
        layer5 = tf.nn.conv2d(layer5, self.weight_variable(
            shape=[1, 1, 32, 1], name='conv_layer5'),
                              strides=[1,1,1,1], padding='SAME', data_format='NCHW')

        return tf.nn.sigmoid(tf.nn.bias_add(layer5, self.bias_variable([1], name='cb_out'), data_format='NCHW'))

    def ResidualBlock(self, x, input_channels=None, kernel_size=None, name=None):
        with tf.name_scope('Residual_block'):
            h_1 = tf.nn.conv2d(x, self.weight_variable(
                shape=[kernel_size, kernel_size, input_channels, input_channels], name='RB1_'+name),
                               strides=[1,1,1,1], padding='SAME', data_format='NCHW')
            h_1_batch_normed = tf.keras.layers.BatchNormalization()(h_1, training=self.is_training)
            h_1_act = tf.nn.relu(h_1_batch_normed)
            h_2 = tf.nn.conv2d(h_1_act, self.weight_variable(
                shape=[kernel_size, kernel_size, input_channels, input_channels], name='RB2_'+name),
                               strides=[1,1,1,1], padding='SAME', data_format='NCHW')
            h_2_batch_normed = tf.keras.layers.BatchNormalization()(h_2, training=self.is_training)
            h_3 = tf.add(h_2_batch_normed, x)
        return tf.nn.relu(h_3)

    def loss(self, y, t):
        mse = tf.reduce_mean(tf.square(y - t))
        return mse

    def training(self, loss):
        optimizer = tf.train.RMSPropOptimizer(0.01)
        #optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
        train_step = optimizer.minimize(loss)
        return train_step


    def fit(self, X_train, Y_train, X_validation, Y_validation, epochs, batch_size):
        x = tf.placeholder(tf.float32, shape=[None, self.channels, self.height, self.width])
        t = tf.placeholder(tf.float32, shape=[None, self.channels, self.height, self.width])
        n_batch = tf.placeholder(tf.int32, shape=[])
        is_training = tf.placeholder(tf.bool)
        self.is_training = is_training
        y = self.inference(x)
        loss = self.loss(y, t)
        train_step = self.training(loss)
        
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session(config=self.config)
        sess.run(init)

        val_batch_size = 256
        n_batches = len(X_train) // batch_size
        val_batches = len(X_validation) // val_batch_size

        #saver.restore(sess, MODEL_DIR + '/model.ckpt')

        ttr_loss = 0
        for i in range(val_batches):
            start = i * val_batch_size
            end = start + val_batch_size
            val_loss = loss.eval(session=sess, feed_dict={
                x: X_validation[start:end],
                t: Y_validation[start:end],
                n_batch: val_batch_size,
                self.is_training: False
            })
            ttr_loss = ttr_loss + val_loss

        f = open('loss_1.1.txt', 'w')
        f.write("epoch: %d train: %e val: %e"
                %(0, 0, ttr_loss/val_batches))

        for epoch in range(epochs):
            s = time.time()
            ttr_loss = 0
            shuffle_num = [i for i in range(len(Y_train))]
            np.random.shuffle(shuffle_num)

            X_train_shuffled=[]
            Y_train_shuffled=[]

            for i in range(1, len(Y_train)):
                X_train_shuffled.append(X_train[shuffle_num[i]])
                Y_train_shuffled.append(Y_train[shuffle_num[i]])

            for i in range(n_batches):
                e = time.time()
                start = i * batch_size
                end = start + batch_size

                _, train_loss = \
                                sess.run([train_step, loss], feed_dict={
                                    x: X_train_shuffled[start:end],
                                    t: Y_train_shuffled[start:end],
                                    n_batch: batch_size,
                                    self.is_training: True
                                })

                ttr_loss += train_loss
                interval = time.time()
                interval = interval - e
                progress_bar(epoch, epochs, i, n_batches, train_loss, val_loss, interval)

            val_losses = 0

            for i in range(val_batches):
                start = i * val_batch_size
                end = start + val_batch_size
                val_loss = loss.eval(session=sess, feed_dict={
                    x: X_validation[start:end],
                    t: Y_validation[start:end],
                    n_batch: val_batch_size,
                    self.is_training: False
                })
                val_losses += val_loss

            interval = time.time()
            interval = interval - s
            progress_bar(epoch, epochs, n_batches-1, n_batches, ttr_loss/n_batches, val_losses/val_batches, interval)
            sys.stdout.write("\n")
            sys.stdout.flush()

            f.write("\n")
            f.write("epoch: %d train: %e val: %e"
                    %(epoch, ttr_loss/n_batches, val_losses/val_batches))
            f.flush()

            if epoch % 10 == 0:
                model_path = saver.save(sess, MODEL_DIR + '/model.ckpt')

        model_path = saver.save(sess, MODEL_DIR + '/model.ckpt')            

        self._sess = sess
        self._x = x
        self._y = y
        self._n_batch = n_batch

        self.evaluate(X_train, X_validation, Y_train, Y_validation)
        sess.close()

    def evaluate(self, X_train, X_validation, Y_train, Y_validation):
        sess = self._sess
        pred = sess.run(self._y, feed_dict={
            self._x: X_train[:10],
            self._n_batch: 10,
            self.is_training: False
        })
        val = sess.run(self._y, feed_dict={
            self._x: X_validation[:10],
            self._n_batch: 10,
            self.is_training: False
        })
        for i in range(10):
            np.savetxt('vp_train_ans_'+str(i)+'.csv', Y_train[i].reshape(64, 128), delimiter=',')
            np.savetxt('vp_train_pred_'+str(i)+'.csv', pred[i].reshape(64,128), delimiter=',')
            np.savetxt('vp_val_ans_'+str(i)+'.csv', Y_validation[i].reshape(64, 128), delimiter=',')
            np.savetxt('vp_val_pred_'+str(i)+'.csv', val[i].reshape(64,128), delimiter=',')

        
def progress_bar(epoch, epochs, n, n_batch, train_loss, val_loss, interval):
    bar = "[" + "#"*int(1+n*20/n_batch) + " "*(int(20) - int(1+n*20/n_batch)) + "]"
    sys.stdout.write("epoch:%d %s(%d/%d) train: %e val: %e time: %e \n"
                     %(epoch, bar ,1+n, n_batch, train_loss, val_loss, interval))
    sys.stdout.flush()                           
    

if __name__ == '__main__':
    print('----------------------------------', flush=True)
    print('----- (C) 2018 R.Yamaguchi--------', flush=True)
    print('----------------------------------', flush=True)

    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "68"

    num = 0
    inp_name = []
    for i in range(1,128,8):
        for j in range(1,2501,1):
            inp_name.append([i,j])
            
    inp_file = np.load('inp.npy')
    uz_r = np.loadtxt('uz.csv', dtype=float)[1:2501]
    
    uz_r = uz_r / (4.0 * math.pi) * 128
    print(np.shape(uz_r))
    inp = np.zeros(shape=[len(inp_file), 64, 128], dtype='float32')
    
    for j in range(128):
        inp[:,:,j] = (inp_file[:,:,2*j] + inp_file[:,:,2*j+1]) * 0.50
        
    #inp_tmp = np.zeros(shape=[len(inp_file), 64, 128], dtype='float32')
    #uu = np.zeros(shape=[64], dtype=float)
    uu = 0
    for n in range(1,len(uz_r)):
        uu += uz_r[n] * 0.020
        for i in range(64):
            inp[n,i,:] = np.roll(inp[n,i,:], - uu.astype(int))
    
    inp = inp/-9.720
    inp = np.where(inp > 1.0, 1.0, 0.0)

    del inp_file
    
    data = []
    target = []
    for i in range(len(inp) - 1):
        if inp_name[i][0] == inp_name[i + 1][0]:
            data.append(inp[i])
            target.append(inp[i + 1])
            
    data = np.array(data).reshape(-1, 1, 64, 128)
    target = np.array(target).reshape(-1, 1, 64, 128)
    
    del inp_name, inp
    
    N_train = int(len(data)*0.9)
    X_train = data[:N_train]
    X_validation = data[N_train:]
    del data
    Y_train = target[:N_train]
    Y_validation = target[N_train:]
    del target
    
    print(np.shape(X_train))
    print(np.shape(Y_train))
    print(np.shape(X_validation))
    print(np.shape(Y_validation))

    '''
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    train_data = train_data.reshape(-1, 1, 28, 28)
    eval_data = eval_data.reshape(-1, 1, 28, 28)
    '''
    
    height = 64
    width = 128
    channels = 1
    batch_size = 256
    epochs = 100
    
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_1.1')
    if os.path.exists(MODEL_DIR) is False:
        os.mkdir(MODEL_DIR)

    config = tf.ConfigProto(inter_op_parallelism_threads=2,
                            intra_op_parallelism_threads=mp.cpu_count())
    
    print('call model', flush=True)
    model = ResCNNPredNet(height, width, channels, config)

    print('call model.fit', flush=True)
    model.fit(X_train, Y_train, X_validation, Y_validation, epochs, batch_size)
    
