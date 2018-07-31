import input_data
import numpy as np

path = r"E:\pythonCode\DeepLearning\Cifar10\cifar10\cifar-10-batches-py"
cifar10 = input_data.load_cifar10(path, one_hot = True)
images = cifar10.images
print("训练集图片：" + str(images.shape))
labels = cifar10.labels
print("训练集类别：" + str(labels.shape))
test_images = cifar10.test.images
print("测试集图片："+ str(test_images.shape))
test_labels = cifar10.test.labels
print("测试集类别："+ str(test_labels.shape))
batch_xs, batch_ys = cifar10.next_batch(batch_size = 500, shuffle = True)
print("batch_xs shape is:" + str(batch_xs.shape))
print("batch_ys shape is:" + str(batch_ys.shape))