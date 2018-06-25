import numpy as np
import scipy
from scipy.misc import imread, imresize
import os
'''
    神经网络实现
    linear -> relu -> linear -> relu .... -> linear -> relu -> linear -> sigmoid
'''
np.random.seed(1)

def sigmoid(z):
    A = 1 / (1 + np.exp(-z))
    cache = z
    return A, cache

def relu(z):
    A = np.maximum(0,z)
    cache = z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def init_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation = "sigmoid"):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    L = len(parameters) // 2
    caches = []
    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation= "sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(1 / m) * np.sum(Y * np.log(AL) + (1- Y) * np.log(1 - AL))
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims= True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def updata_parameters(parameters, grads, learning_rate = 0.01):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters

def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))
    probs, caches = L_model_forward(X, parameters)
    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))
    return p

def load_dataSet(file1, file2, n_pl, n_ph):
    #正例
    rootdir = file1
    images = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    X = np.zeros((n_pl*n_ph*3, 1))
    y = []
    for i in range(0,len(images)):
        path = os.path.join(rootdir,images[i])
        if os.path.isfile(path):
            image = imresize(imread(path), (n_pl,n_ph)).reshape(1, -1).T
            image_flatten = image / 255
            X = np.c_[X, image_flatten]
            y.append(1)
    #反例
    rootdir = file2
    images = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(images)):
        path = os.path.join(rootdir,images[i])
        if os.path.isfile(path):
            image = imresize(imread(path), (n_pl, n_ph)).reshape(1, -1).T
            image_flatten = image / 255
            X = np.c_[X, image_flatten]
            y.append(0)
    return X[:, 1:].reshape(n_pl*n_ph*3, -1), np.array(y).reshape(1, -1)