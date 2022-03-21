import numpy as np
#import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#import tensorflow_datasets
#mnist = tensorflow_datasets.load('mnist')

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
#tf.disable_v2_behavior()

import tensorflow_datasets as tfds


class MyDS(object):
    class SubDS(object):
        import numpy as np
        def __init__(self, ds, *, one_hot):
            np = self.__class__.np
            self.ds = [e for e in ds.as_numpy_iterator()]
            self.sds = {(k + 's'): np.stack([
                (e[k] if len(e[k].shape) > 0 else e[k][None]).reshape(-1) for e in self.ds
            ], 0) for k in self.ds[0].keys()}
            self.one_hot = one_hot
            if one_hot is not None:
                self.max_one_hot = np.max(self.sds[one_hot + 's'])

        def _to_one_hot(self, a, maxv):
            np = self.__class__.np
            na = np.zeros((a.shape[0], maxv + 1), dtype=a.dtype)
            for i, e in enumerate(a[:, 0]):
                na[i, e] = True
            return na

        def _apply_one_hot(self, key, maxv):
            assert maxv >= self.max_one_hot, (maxv, self.max_one_hot)
            self.max_one_hot = maxv
            self.sds[key + 's'] = self._to_one_hot(self.sds[key + 's'], self.max_one_hot)

        def next_batch(self, num=16):
            np = self.__class__.np
            idx = np.random.choice(len(self.ds), num)
            res = {k: np.stack([
                (self.ds[i][k] if len(self.ds[i][k].shape) > 0 else self.ds[i][k][None]).reshape(-1) for i in idx
            ], 0) for k in self.ds[0].keys()}
            if self.one_hot is not None:
                res[self.one_hot] = self._to_one_hot(res[self.one_hot], self.max_one_hot)
            for i, (k, v) in enumerate(list(res.items())):
                res[i] = v
            return res

        def __getattr__(self, name):
            if name not in self.__dict__['sds']:
                return self.__dict__[name]
            return self.__dict__['sds'][name]

    def __init__(self, name, *, one_hot=None):
        self.ds = tfds.load(name)
        self.sds = {}
        for k, v in self.ds.items():
            self.sds[k] = self.__class__.SubDS(self.ds[k], one_hot=one_hot)
        if one_hot is not None:
            maxh = max(e.max_one_hot for e in self.sds.values())
            for e in self.sds.values():
                e._apply_one_hot(one_hot, maxh)

    def __getattr__(self, name):
        if name not in self.__dict__['sds']:
            return self.__dict__[name]
        return self.__dict__['sds'][name]


# Get the MNIST data
mnist = MyDS('mnist', one_hot='label')  # tensorflow_datasets.load('mnist')
tf.disable_eager_execution()
tf.disable_v2_behavior()


images = tf.placeholder(tf.float32, [None, 28, 28, 1])
labels = tf.placeholder(tf.float32, [None, 28, 28, 1])


# Encoder
with tf.name_scope('en-conv1'):
    conv1 = tf.layers.conv2d(images, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='SAME', use_bias=True,
                             activation=tf.nn.leaky_relu, name='conv1')
# 28x28x32
with tf.name_scope('en-conv2'):
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='SAME', use_bias=True,
                             activation=tf.nn.leaky_relu, name='conv2')
# 28x28x64
with tf.name_scope('en-pool2'):
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), name='pool2')

# 14x14x64
with tf.name_scope('en-conv3'):
    conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='SAME', use_bias=True,
                             activation=tf.nn.leaky_relu, name='conv3')
# 14x14x64
with tf.name_scope('en-pool1'):
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), name='encoding')

# 7x7x64
# latent space

# Decoder
with tf.name_scope('decoder'):
    conv4 = tf.layers.conv2d(encoded, filters=64, kernel_size=(5, 5), strides=(1, 1), name='conv4', padding='SAME',
                             use_bias=True, activation=tf.nn.leaky_relu)
    # Now 7x7x64
    upsample1 = tf.layers.conv2d_transpose(conv4, filters=64, kernel_size=5, padding='same', strides=2,
                                           name='upsample1')
    # Now 14x14x64
    upsample2 = tf.layers.conv2d_transpose(upsample1, filters=64, kernel_size=5, padding='same', strides=2,
                                           name='upsample2')
    # Now 28x28x64
    logits = tf.layers.conv2d(upsample2, filters=1, kernel_size=(5, 5), strides=(1, 1), name='logits', padding='SAME',
                              use_bias=True)
    # Now 28x28x1
    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.sigmoid(logits, name='recon')


loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
residual_error = tf.subtract(decoded, images)
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)  # cost
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # optimizer

# Training
sess = tf.Session()
saver = tf.train.Saver()
loss = []
valid_loss = []
display_step = 1
epochs = 100
batch_size = 75
lr = 1e-5
#total_batch = 938
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./graphs', sess.graph)
for e in range(epochs):
    #total_batch = int(mnist.train.num_examples / batch_size)
    #total_batch = int(mnist.splits['train'].num_examples // batch_size)
    #for ibatch in range(total_batch):
    for ibatch in range(batch_size):
        batch_x = mnist.train.next_batch(batch_size)
        batch_test_x = mnist.test.next_batch(batch_size)
        #batch_x = info.train.next_batch(batch_size)
        #batch_test_x = info.test.next_batch(batch_size)
        imgs_test = batch_x[0].reshape((-1, 28, 28, 1))
        noise_factor = 0.5
        x_test_noisy = imgs_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs_test.shape)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        imgs = batch_x[0].reshape((-1, 28, 28, 1))
        x_train_noisy = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        batch_cost, _ = sess.run([cost, opt], feed_dict={images: x_train_noisy,
                                                         labels: imgs, learning_rate: lr})

        batch_cost_test = sess.run(cost, feed_dict={images: x_test_noisy, labels: imgs_test})
    if (e + 1) % display_step == 0:
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost), "Validation loss: {:.4f}".format(batch_cost_test))

    print("appending loss")
    loss.append(batch_cost)
    valid_loss.append(batch_cost_test)
    if e+1 == epochs:
        plt.plot(range(e + 1), loss, 'bo', label='Training loss')
        plt.plot(range(e + 1), valid_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
        plt.figure()
        plt.show()
    #saver.save(sess, './encode_model')

#batch_x = info.test.next_batch(10)
batch_x = mnist.test.next_batch(10)
imgs = batch_x[0].reshape((-1, 28, 28, 1))
x_test_noisy = imgs + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

recon_img = sess.run([decoded], feed_dict={images: x_test_noisy})[0]
res_err = sess.run([residual_error], feed_dict={images: imgs})[0]
pca = PCA(n_components=5)
pca.fit(res_err)
reserr_pca = pca.transform(res_err)

# plt.figure(figsize=(20, 4))
# plt.title('Reconstructed Images')
plt.figure(figsize=(20, 4))
toPlot = (imgs, x_test_noisy, recon_img, res_err)
for i in range(10):
    for j in range(4):
        ax = plt.subplot(4, 10, 10*j+i+1)
        plt.imshow(toPlot[j][i, :].reshape(28, 28), interpolation="nearest", vmin=0, vmax=1, cmap='gray')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.tight_layout()
print("Original Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.title('Reconstructed Images')
    plt.imshow(imgs[i, ..., 0], cmap='gray')
#plt.show()
plt.figure(figsize=(20, 4))
print("Noisy Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.title('Noisy Images')
    plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))
print("Reconstruction of Noisy Images")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.title('Reconstruction of Noisy Images')
    plt.imshow(recon_img[i, ..., 0], cmap='gray')
# plt.show()

# plt.figure(figsize=(20, 4))
print("Mean residual error")
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.title('Mean residual error')
    plt.imshow(res_err[i, ..., 0], cmap='gray')

plt.show()

writer.close()
sess.close()
