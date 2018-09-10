import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def generate_net(input_z):
    with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
            # in_size = input_data.shape[1]
            output = tf.layers.dense(input_z, 128, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.layers.dense(output, 784, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            output = tf.nn.sigmoid(logits)
            return output 

def discriminator_net(X):
    with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
        y_pred = tf.layers.dense(X, 256, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        y_pred = tf.nn.dropout(y_pred, keep_prob=0.5)
        y_pred = tf.layers.dense(y_pred, 64, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        logits = tf.layers.dense(y_pred, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
        out = tf.nn.sigmoid(logits)
        return out, logits

def model_loss(D_real, D_fake):
    # D_loss = -tf.reduce_mean(tf.log(D_real + 1e-6) + tf.log(1. - D_fake + 1e-6))
    # G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-6))
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
    return D_loss, G_loss
def train(learning_rate, batch_size, k, verbose=True, plot_loss_history=True):
    Z = tf.placeholder(tf.float32, [None, 100])
    X = tf.placeholder(tf.float32, [None, 784])

    G_sample = generate_net(Z)
    D_real, D_logits_real = discriminator_net(X)
    D_fake, D_logits_fake = discriminator_net(G_sample)

    #loss
    D_loss, G_loss = model_loss(D_logits_real, D_logits_fake)
    #optimize
    var_all = tf.trainable_variables()
    d_var = [var for var in var_all if var.name.startswith("discriminator")]
    g_var = [var for var in var_all if var.name.startswith("generator")]
    D_optimize = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_var)
    G_optimize = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_var)

    d_loss_history = []
    g_loss_history = []
    minst = input_data.read_data_sets('MNIST_data', one_hot=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(sess.run(D_fake,feed_dict={Z:sampe_z(batch_size,100)}))
        plt.figure()
        for it in range(100000):
            if it % 1000 == 0:
                samples = sess.run(G_sample, feed_dict={
                           Z: sample_z(16, 100)})  # 16*784
                fig = plot(samples)
                fig.savefig("./gan1/" + str(it / 1000) + ".jpg")
                
                # plt.show(fig)
                plt.close(fig)
            X_real, _ = minst.train.next_batch(batch_size)
            for d in range(k):
                _, D_current_loss = sess.run([D_optimize, D_loss], feed_dict={Z: sample_z(batch_size,100), X: X_real})
            _, G_current_loss = sess.run([G_optimize, G_loss], feed_dict={Z:sample_z(batch_size,100)})
            d_loss_history.append(D_current_loss)
            g_loss_history.append(G_current_loss)    
            if it % 1000 == 0 and verbose == True:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'.format(D_current_loss))
                print('G_loss: {:.4}'.format(G_current_loss))            
                if plot_loss_history == True:     
                    plt.ion()
                    plt.plot(np.arange(0, len(d_loss_history)), d_loss_history, color='b')
                    plt.plot(np.arange(0, len(g_loss_history)), g_loss_history, color='g')
                    plt.pause(0.1)      
                    plt.cla()
def sample_z(m,n):
    return np.random.uniform(-1., 1.,size=[m,n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    for i, sample in enumerate(samples):
        ax = plt.subplot(4,4,i+1)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
if __name__ == "__main__":
    train(0.001, 128, k=1, plot_loss_history=True)

    