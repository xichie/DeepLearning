# coding: utf-8

from keras import layers
import keras.backend as K
from keras.datasets import mnist
from keras import models
from keras.losses import binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt

# 加载mnist数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
image_size = X_train.shape[1]

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = np.reshape(X_train, (-1, np.prod(X_train.shape[1:])))
X_test = np.reshape(X_test, (-1, np.prod(X_test.shape[1:])))
print("X_train shape: ",X_train.shape)	# (60000, 784)
print("X_test shape: ",X_test.shape)	# (10000, 784)

# 网络参数
epochs = 50
batch_size = 64
latent_dim = 2
original_dim = image_size * image_size
intermediate_dim = 512
input_shape = (original_dim,)


# 参数技巧
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + epsilon * K.exp(z_log_var * 0.5)


# 编码器
inputs = layers.Input(shape=input_shape, name = 'encoder_input')
x = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim, name = 'z_mean')(x)
z_log_var = layers.Dense(latent_dim, name = 'z_log_var')(x)

z = layers.Lambda(sampling, output_shape=(latent_dim,), name = 'z')([z_mean, z_log_var])
encoder = models.Model(inputs, [z_mean, z_log_var, z], name='vae_encoder')
encoder.summary()

# 解码器
latent_inputs = layers.Input(shape=(latent_dim,), name = 'decoder_inputs')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)

decoder = models.Model(latent_inputs, outputs, name='vae_decoder' )
decoder.summary()

# vae网络
outputs = decoder(encoder(inputs)[2])
vae = models.Model(inputs, outputs, name='vae')
vae.summary()

reconstruction_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=-1)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss *= -0.5
kl_loss = K.sum(kl_loss, axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', metrics=['acc'])

history = vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, None))

vae.save('./vae_model.h5')


# 可视化结果
n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = n * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
plt.show()

