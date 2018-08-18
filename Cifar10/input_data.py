import pickle
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave

class Cifar10:
    class test:
        pass
    def __init__(self, path, one_hot = True):
        self.path = path
        self.one_hot = one_hot
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 50000
        
    
    def _load_data(self):
        dic = {}
        images = np.zeros([10000, 3,32,32])
        labels = []
        files = os.listdir(self.path)
        for file in files:
            if re.match("data_batch_*", file):          
                with open(self.path +"\\" + file, 'rb') as fo:       #load train data
                    dic = pickle.load(fo, encoding = 'bytes')
                    images = np.r_[images, dic[b"data"].reshape([-1,3,32,32])]
                    labels.append(dic[b"labels"])
            elif re.match("test_batch", file):          #load test data 
                with open(self.path + "\\" + file , 'rb') as fo:
                    dic = pickle.load(fo, encoding = 'bytes')
                    test_images = np.array(dic[b"data"].reshape([-1, 3, 32, 32]))
                    test_labels = np.array(dic[b"labels"])
        dic["train_images"] = images[10000:].transpose(0,2,3,1)
        dic["train_labels"] = np.array(labels).reshape([-1, 1]) 
        dic["test_images"] = test_images.transpose(0,2,3,1)
        dic["test_labels"] = test_labels.reshape([-1, 1])
        if self.one_hot == True:
            dic["train_labels"] = self._one_hot(dic["train_labels"], 10)
            dic["test_labels"] = self._one_hot(dic["test_labels"], 10)

        self.images, self.labels =  dic["train_images"], dic["train_labels"]
        self.test.images, self.test.labels = dic["test_images"], dic["test_labels"]
        return [dic["train_images"], dic["train_labels"], dic["test_images"], dic["test_labels"] ]


    def next_batch(self, batch_size,shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch
                images_new_part = self._images[start:end]
                labels_new_part = self._labels[start:end]
                return np.concatenate(
                    (images_rest_part, images_new_part), axis=0), np.concatenate(
                        (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

    def _one_hot(self, labels, num):
        size= labels.shape[0]
        label_one_hot = np.zeros([size, num])
        for i in range(size):
            label_one_hot[i, np.squeeze(labels[i])] = 1
        return label_one_hot

def load_cifar10(path, one_hot = True):
    cifar10 = Cifar10(path, one_hot)
    cifar10._load_data()
    return cifar10

if __name__ == "__main__":
    path = r"E:\pythonCode\TensorFlow\cifar10\cifar-10-batches-py"
    cifar10 = load_cifar10(path, one_hot = False)
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

    #plot image
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    num_classes = len(classes)
    samples_per_class = 7
    for y, clss in enumerate(classes):
        idxs = np.flatnonzero(labels == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(images[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(clss)
    plt.show()




    ## save image
    # images = images.reshape(-1, 3, 32, 32).astype('uint8')          #要转化为uint8才能正常可视化
    # for i in range(images.shape[0]):
    #     imgs = images[i]
    #     if i < 100:#只循环100张图片,这句注释掉可以便利出所有的图片,图片较多,可能要一定的时间
    #         img0 = imgs[0]
    #         img1 = imgs[1]
    #         img2 = imgs[2]
    #         i0 = Image.fromarray(img0)
    #         i1 = Image.fromarray(img1)
    #         i2 = Image.fromarray(img2)
    #         img = Image.merge("RGB",(i0,i1,i2))
    #         name = "img" + str(i)
    #         img.save(r"E:\pythonCode\TensorFlow\cifar10\images\\" + name + ".png")#文件夹下是RGB融合后的图像

    # print ("保存完毕.")