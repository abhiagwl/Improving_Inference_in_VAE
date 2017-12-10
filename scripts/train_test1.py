import tensorflow as tf
import numpy as np


visualize = True
batch_size = 100
hidden_dim = 30
h1 = 500
h2 = 300
learning_rate = 0.0005
max_epochs = 100


train = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_train.amat').readlines()]).astype('float32')
valid = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_valid.amat').readlines()]).astype('float32')

#------------------------------------- Model ---------------------------------
x = tf.placeholder(tf.float32, shape=[None, 28*28])
phase_train = tf.placeholder(tf.bool)

with tf.name_scope("encoder"):
    ffd1 = tf.layers.dense(x, units=h1, activation=tf.nn.relu,use_bias=True, kernel_initializer = tf.contrib.layers.xavier_initializer() , bias_initializer = tf.constant_initializer(0.1))
    
    ffd2 = tf.layers.dense(ffd1, units=h2, activation=tf.nn.relu,use_bias=True, kernel_initializer = tf.contrib.layers.xavier_initializer() , bias_initializer = tf.constant_initializer(0.1))
    
    # ffd3 = tf.layers.dense(x, units=500, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())

	# conv1 = tf.layers.conv2d(inputs=tf.reshape(x, [-1, 28, 28, 1]), 
	# 			filters=32, kernel_size=[5, 5], kernel_initializer = tf.contrib.layers.xavier_initializer(),
	# 			padding="same", activation=None)
	# conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=phase_train))
	# pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=(2, 2))
	
# 	conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_initializer = tf.contrib.layers.xavier_initializer(),
# 				kernel_size=[5, 5], padding="same", activation=None)
# 	conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=phase_train))
# 	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=(2, 2))

# 	flat = tf.reshape(pool2, [-1, 32 * 7 * 7])
    mu = tf.layers.dense(ffd2, units=hidden_dim, activation=None, kernel_initializer = tf.contrib.layers.xavier_initializer())
    sd = tf.layers.dense(ffd2, units=hidden_dim, activation=tf.nn.softplus, kernel_initializer = tf.contrib.layers.xavier_initializer())

    eps = tf.contrib.distributions.MultivariateNormalDiag(loc=[0.]*hidden_dim, scale_diag=[1.]*hidden_dim)
    samples = mu + eps.sample(tf.shape(sd)[0]) * sd

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# REAL NVP
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
with tf.name_scope("decoder"):
    ffd1_d = tf.layers.dense(samples, units=h1, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),bias_initializer = tf.constant_initializer(0.1))
    
    ffd2_d = tf.layers.dense(ffd1_d, units=h2, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),bias_initializer = tf.constant_initializer(0.1))
	# flat = tf.nn.relu(tf.layers.batch_normalization(flat, training=phase_train))
# 	deconv1 = tf.layers.conv2d_transpose(tf.reshape(flat, [-1, 7, 7, 32]), filters=32, kernel_size=[5,5], strides=(2, 2), padding='same', 
# 						kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)
# 	deconv1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=phase_train))
	
# 	deconv2 = tf.layers.conv2d_transpose(deconv1, filters=32, kernel_size=[5,5], strides=(2, 2), padding='same', 
# 						kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)
# 	deconv2 = tf.nn.relu(tf.layers.batch_normalization(deconv2, training=phase_train))
	
	# out = tf.layers.conv2d_transpose(deconv2, filters=1, kernel_size=[5,5], strides=(1, 1), padding='same', 
	# 					kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)
    out = tf.layers.dense(ffd2_d, units = 28*28 , activation = None, kernel_initializer = tf.contrib.layers.xavier_initializer())

reconstructed_image = tf.nn.sigmoid(out) #> 0.5
# likelihood = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(x, [-1, 28, 28, 1]), logits=out))

likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=out),1)

likelihood_ = tf.reduce_mean(likelihood)

loss =  tf.reduce_mean(likelihood - tf.reduce_sum(0.5 * (1.+ tf.log(sd ** 2) - mu ** 2 - sd ** 2), 1))

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(max_epochs):
    np.random.shuffle(train)
    for i in range(train.shape[0]/batch_size):
        batch = train[i*batch_size : (i+1)*batch_size, :]
        tr_lik, _ = sess.run([likelihood_, train_step], feed_dict={x:batch, phase_train:True})
        # print tr_lik
    val_lik = sess.run([likelihood_], feed_dict={x:valid, phase_train:False})
    print "validation : " + str(val_lik)
if visualize:
    x_sample = np.random.shuffle(test)[:100]
    x_reconstruct = sess.run([recontructed_image], feed_dict={x: x_sample,phase_train = False})
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()


