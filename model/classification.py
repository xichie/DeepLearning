import numpy as np
from nn_utils import *
from nn_utils import load_dataSet

X, y =load_dataSet(r"E:\python\vscode\images1", r"E:\python\vscode\images2",64,64)
print(X.shape)
print(y)
def model(X_train, y_train, num_iterator=1000, learning_rate = 0.01):
    n_x, m = X_train.shape
    layer_dims = [n_x, 50, 1]
    parameters = init_parameters(layer_dims)
    costs = []
    for i in range(num_iterator):
        AL, caches = L_model_forward(X_train, parameters)
        cost = compute_cost(AL, y_train)
        grads = L_model_backward(AL, y_train, caches)
        updata_parameters(parameters, grads, learning_rate)
        costs.append(cost)
        if i%100 == 0:
            print("iterator of " + str(i) + "cost:" + str(cost))
    return parameters, costs

parameters, costs = model(X, y)
predict(X, y,parameters)

import matplotlib.pyplot as plt
plt.plot(np.arange(0, 1000),costs)
plt.show()


##test
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

fname = r"C:\Users\qjx\Desktop\test1.jpg"
image = imresize(imread(fname, flatten=False), (64,64))
my_image = image.reshape(1, -1).T
my_image = my_image / 255
plt.imshow(image)
plt.show()

p = predict(my_image, parameters)
print("It's juhua:" + str(p))
