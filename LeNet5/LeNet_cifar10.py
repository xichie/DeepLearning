import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data
from tqdm import tqdm

class LeNet5:
    def __init__(self):
        pass

    def inference(self, input_tensor):
        with tf.variable_scope("layer1-conv1"):
            conv1_weight = tf.get_variable(name = "conv1_variable", shape=[5,5,3,64], initializer=tf.truncated_normal_initializer()) 
            conv1_bias = tf.get_variable(name = "conv1_bias", shape = [64], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(input = input_tensor, filter = conv1_weight, strides = [1, 1, 1, 1], padding = "SAME")
            relu1 = tf.nn.relu(tf.add(conv1, conv1_bias))
            pool1 = tf.nn.max_pool(relu1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
            
        with tf.variable_scope("layer2-conv2"):
            conv2_weight = tf.get_variable(name = "conv2_variable", shape=[5,5,64,128], initializer=tf.truncated_normal_initializer()) 
            conv2_bias = tf.get_variable(name = "conv2_bias", shape = [128], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(input = pool1, filter = conv2_weight, strides = [1, 1, 1, 1], padding = "SAME")
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
            pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")

        with tf.variable_scope("layer3-fc1"):
            conv_layer_flatten = tf.layers.flatten(inputs = pool2)      #[batch_size, dim]
            dim = conv_layer_flatten.get_shape()[1]
            fc1_variable = tf.get_variable(name = 'fc1_variable', shape = [dim, 256], initializer = tf.random_normal_initializer()) * 0.01
            fc1_bias = tf.get_variable(name = 'fc1_bias', shape = [1, 256], initializer = tf.constant_initializer(value=0.0))
            fc1 = tf.nn.relu(tf.add(tf.matmul(conv_layer_flatten, fc1_variable), fc1_bias))     #[batch_size, 256]

        with tf.variable_scope("layer4-fc2"):
            fc2_variable = tf.get_variable(name = "fc2_variable", shape=[256,84], initializer=tf.random_normal_initializer())  * 0.01 #[batch_size, 84]
            fc2_bias = tf.get_variable(name = "fc2_bias", shape=[1, 84],initializer = tf.constant_initializer(value=0))
            fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, fc2_variable), fc2_bias))                    #[batch_size, 84]
            # fc2 = tf.nn.dropout(fc2, keep_prob = 0.5)

        with tf.variable_scope("layer5-output"):
            output_variable = tf.get_variable(name = "output_variable", shape = [84, 10],initializer = tf.random_normal_initializer()) * 0.01
            output_bias = tf.get_variable(name = "output_bias", shape = [1, 10],initializer = tf.constant_initializer(value=0.0))
            output = tf.add(tf.matmul(fc2, output_variable), output_bias)        #[batch_size, 10]

        return output

    def train(self, data, epochs = 10, batch_size = 32, learning_rate = 0.001, save_model=True):
        costs = []
        loss = 0
        x = tf.placeholder(dtype = tf.float32, shape = [batch_size, 32, 32, 3], name = "x")
        y = tf.placeholder(dtype = tf.float32, shape = [batch_size, 10], name = "y")
        output = self.inference(x)
        # cross_entropy = -tf.reduce_sum(y * tf.log(y_pred) + (1 - y) * tf.log(1 - y_pred)) / batch_size
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y), name = "loss")
        train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
        saver = tf.train.Saver()
       
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(epochs):
                for _ in tqdm(range(60000 // batch_size), desc=str("Epoch  " + str(epoch) + "/" + str(epochs))):
                    batch_xs, batch_ys = data.next_batch(batch_size)
                    batch_xs /= 255                                             #normlaize
                    loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y: batch_ys})
                print("Epoch %d / %d: Train loss:%f "%(epoch, epochs, loss))
                costs.append(loss)
            if save_model == True:            
                saver.save(sess, r"./cifar10Model/model.ckpt")
        plt.figure()
        plt.title("loss")
        plt.xlabel("Epochs")
        plt.xlabel("Loss")
        plt.plot(np.arange(1, epochs + 1), costs)
        plt.savefig(r'./cifar10Model/history.png')

    def evaluate(self, images, y_true, batch_size= 32):
        num_images = images.shape[0]
        true_count = 0
        tf.reset_default_graph()
        x = tf.placeholder(dtype = tf.float32, shape=[None, 32,32,3])
        output = self.inference(x)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, r"./cifar10Model/model.ckpt")
            for batch in tqdm(range(num_images// batch_size)):
                x_batch = images[batch_size * batch: batch_size*(batch + 1)]
                y_batch = y_true[batch_size * batch: batch_size*(batch + 1)]
                y_pred = np.argmax(sess.run(tf.nn.softmax(output), feed_dict={x: x_batch / 255.}), axis = 1).reshape(-1, 1)
                true_count += np.sum(y_pred == y_batch)
            if batch_size * (num_images // batch_size) < num_images:
                x_batch = images[batch_size * (num_images // batch_size): ]
                y_batch = y_true[batch_size * (num_images // batch_size): ]
                y_pred = np.argmax(sess.run(tf.nn.softmax(output), feed_dict={x: x_batch / 255.}), axis = 1).reshape(-1, 1)
                true_count += np.sum(y_pred == y_batch)
            accuracy = true_count / num_images 
            print("accuracy is " + str(accuracy))
        return accuracy



if __name__ == "__main__":
    cifar10 = input_data.load_cifar10(r"E:\pythonCode\TensorFlow\cifar10\cifar-10-batches-py")   
    model = LeNet5()
    print("Training......")
    model.train(cifar10, 10, 32, learning_rate=0.001)
    print("Finish!")

    # model.evaluate(cifar10.images, cifar10.labels)
    X_test, y_test = cifar10.test.images, cifar10.test.labels
    y_test = np.argmax(y_test, axis = 1).reshape(-1, 1)
    test_acc = model.evaluate(X_test, y_test, batch_size=32)
    print(test_acc)