import tensorflow as tf
import numpy as np

#-------------------------------- Hyper Parameters -----------------------------
batch_size = 100
hidden_dim = 30
learning_rate = 0.0005
max_epochs = 100

#-------------------------------- Loading data -----------------------------

train = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_train.amat').readlines()]).astype('float32')
valid = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_valid.amat').readlines()]).astype('float32')
test = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_test.amat').readlines()]).astype('float32')

#-------------------------------- Placeholders -----------------------------

x = tf.placeholder(tf.float32, shape=[None, 28*28])
phase_train = tf.placeholder(tf.bool)
t = tf.placeholder(tf.float32)
# z = tf.placeholder(tf.float32 , shape=[None,hidden_dim])
# gen_image = tf.placeholder(tf.bool)

#-------------------------------- norm layer -----------------------------

def weight_variable(shape, name=None):
    if name:
        w = tf.truncated_normal(shape, stddev=0.1, name=name)
    else:
        w = tf.truncated_normal(shape, stddev=0.1)

    return w


def bias_variable(shape, name=None):
    # avoid dead neurons
    if name:
        b = tf.constant(0.1, shape=shape, name=name)
    else:
        b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)
def norm_layer(z):
    w = weight_variable([hidden_dim,1])
    b = bias_variable([1,])
    u = weight_variable([hidden_dim,1])
    
    m = tf.matmul(z,w) + b
    h = tf.nn.tanh(m)
    
    out1 = z + tf.matmul(h,u,transpose_b = True)
    
    h_ = tf.gradients(h,m)[0]
    phi = tf.matmul(h_,w,transpose_b = True)
    out2 = tf.log(tf.abs(1 + tf.matmul(phi,u)))
    
    return out1, out2
#---------------------------------------- Encoder -------------------------------------------


with tf.name_scope("encoder"):
	conv1 = tf.layers.conv2d(inputs=tf.reshape(x, [-1, 28, 28, 1]), 
				filters=32, kernel_size=[5, 5], kernel_initializer = tf.contrib.layers.xavier_initializer(),
				padding="same", activation=None)
	conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=phase_train))
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=(2, 2))
	
	conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_initializer = tf.contrib.layers.xavier_initializer(),
				kernel_size=[5, 5], padding="same", activation=None)
	conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=phase_train))
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=(2, 2))

	flat = tf.reshape(pool2, [-1, 32 * 7 * 7])
	mu = tf.layers.dense(flat, units=hidden_dim, activation=None, kernel_initializer = tf.contrib.layers.xavier_initializer())
	sd = tf.layers.dense(flat, units=hidden_dim, activation=tf.nn.softplus, kernel_initializer = tf.contrib.layers.xavier_initializer())

	eps = tf.contrib.distributions.MultivariateNormalDiag(loc=[0.]*hidden_dim, scale_diag=[1.]*hidden_dim)
	samples = mu + eps.sample(tf.shape(sd)[0]) * sd

#-------------------------------------- Normalizing Flow ----------------------

with tf.name_scope("norm_flow_1"):
    input_z1 = samples
    z1, pz1 = norm_layer(input_z1)

with tf.name_scope("norm_flow_2"):
    input_z2 = z1
    z2, pz2 = norm_layer(input_z2)
    

#-------------------------------------- Decoder -----------------------------------


with tf.name_scope("decoder"):
    # input_z = tf.where(gen_image,z,samples)
    # input_z = samples
    input_z = z2
    flat = tf.layers.dense(input_z, units=32 * 7 * 7, activation=None, kernel_initializer = tf.contrib.layers.xavier_initializer())
    flat = tf.nn.relu(tf.layers.batch_normalization(flat, training=phase_train))
    deconv1 = tf.layers.conv2d_transpose(tf.reshape(flat, [-1, 7, 7, 32]), filters=32, kernel_size=[5,5], strides=(2, 2), padding='same', 
                        kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)
    deconv1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=phase_train))

    deconv2 = tf.layers.conv2d_transpose(deconv1, filters=32, kernel_size=[5,5], strides=(2, 2), padding='same', 
                        kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)
    deconv2 = tf.nn.relu(tf.layers.batch_normalization(deconv2, training=phase_train))

    out = tf.layers.conv2d_transpose(deconv2, filters=1, kernel_size=[5,5], strides=(1, 1), padding='same', 
                        kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)

#-------------------------------------- Reconstruction and losses -----------------------------------

reconstructed_image = tf.nn.sigmoid(out) > 0.5

likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(x, [-1, 28, 28, 1]), logits=out),axis=[1,2,3])

likelihood_ = tf.reduce_mean(likelihood)

# loss =  tf.reduce_mean(likelihood - tf.reduce_sum(0.5 * (1.+ tf.log(sd ** 2) - mu ** 2 - sd ** 2), axis=1))
# aplha_t = tf.constant(0.01,dtype= tf.float32)
# beta_t = tf.minimum(1.,0.01 + t/10000)
loss =  tf.reduce_mean(likelihood) -0.5*tf.reduce_mean(tf.reduce_sum(tf.log(sd),axis =1)) + tf.reduce_mean(tf.reduce_sum(0.5*input_z**2 ,axis=1))-tf.reduce_mean(pz1 + pz2)


#-------------------------------------- Updates for batch norm and other layers -----------------------------------

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#-------------------------------------- Clearning GPU space after script is done -----------------------------------

gpu_options = tf.GPUOptions()#per_process_gpu_memory_fraction=0.5)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# sess = tf.Session(config=config)

# sess = tf.Session()
sess.run(tf.global_variables_initializer())

#-------------------------------------- Training -----------------------------------
iteration = 0
for _ in range(max_epochs):
    np.random.shuffle(train)
    for i in range(train.shape[0]/batch_size):
        iteration+=1
        batch = train[i*batch_size : (i+1)*batch_size, :]
        tr_lik, _ = sess.run([likelihood_, train_step], feed_dict={x:batch, phase_train:True,t:(iteration+1)})
                                                                   #, z:np.zeros((batch_size,hidden_dim), gen_image:False) })
        # print tr_lik
    test_lik,test_loss = sess.run([likelihood_,loss], feed_dict={x:test, phase_train:False,t:0})
    print "test likelihood: " + str(test_lik)
    print "test loss: " + str(test_loss)
    

    val_lik,val_loss = sess.run([likelihood_,loss], feed_dict={x:valid, phase_train:False,t:0})
    print "validation likelihood : " + str(val_lik)
    print "validation loss: " + str(val_loss)

