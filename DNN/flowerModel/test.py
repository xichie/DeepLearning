import numpy as np
from nn_utils import *
import scipy
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import json

def predict(X, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))
    probs, caches = L_model_forward(X, parameters)
    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    return p


fname = r"E:\python\vscode\images1\timg (1).jpg"
image = imresize(imread(fname, flatten=False), (64,64))
my_image = image.reshape(1, -1).T
my_image = my_image / 255
print(my_image.shape)
# plt.imshow(image)
# plt.show()


