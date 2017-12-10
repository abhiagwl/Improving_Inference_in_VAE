import tensorflow as tf
import numpy as np


visualize = True
batch_size = 100
hidden_dim = 30
h1 = 500
h2 = 300
learning_rate = 0.0005
max_epochs = 10

summaries_dir = "../data/logs"

train = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_train.amat').readlines()]).astype('float32')
valid = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_valid.amat').readlines()]).astype('float32')
test = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_test.amat').readlines()]).astype('float32')

#---------------------------------------------------------------------------------------------------------#
x = tf.placeholder(tf.float32, shape=[None, 28*28])
phase_train = tf.placeholder(tf.bool)
z = tf.placeholder(tf.float32 , shape=[None,hidden_dim])
gen_image = tf.placeholder(tf.bool)

#-------------------------------- Utility functions -----------------------------


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

        
#---------------------------- Encoder ------------------------------



with tf.name_scope("encoder"):
    ffd1 = tf.layers.dense(x, units=h1, activation=tf.nn.relu,use_bias=True, kernel_initializer = tf.contrib.layers.xavier_initializer() , bias_initializer = tf.constant_initializer(0.1))
    
    ffd2 = tf.layers.dense(ffd1, units=h2, activation=tf.nn.relu,use_bias=True, kernel_initializer = tf.contrib.layers.xavier_initializer() , bias_initializer = tf.constant_initializer(0.1))
    
    mu = tf.layers.dense(ffd2, units=hidden_dim, activation=None, kernel_initializer = tf.contrib.layers.xavier_initializer())
    sd = tf.layers.dense(ffd2, units=hidden_dim, activation=tf.nn.softplus, kernel_initializer = tf.contrib.layers.xavier_initializer())

    eps = tf.contrib.distributions.MultivariateNormalDiag(loc=[0.]*hidden_dim, scale_diag=[1.]*hidden_dim)
    samples = mu + eps.sample(tf.shape(sd)[0]) * sd

#---------------------------- Normalizing ------------------------------

# REAL NVP


#---------------------------- decoder ------------------------------

with tf.name_scope("decoder"):
#     if gen_image is True:
#         input_z = z
#     else:
#         input_z = samples
    input_z = tf.where(gen_image,z,samples)

    ffd1_d = tf.layers.dense(input_z, units=h1, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),bias_initializer = tf.constant_initializer(0.1))
    
    ffd2_d = tf.layers.dense(ffd1_d, units=h2, activation=tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),bias_initializer = tf.constant_initializer(0.1))
    out = tf.layers.dense(ffd2_d, units = 28*28 , activation = None, kernel_initializer = tf.contrib.layers.xavier_initializer())

#---------------------------- Reconstruction and losses ------------------------------

reconstructed_image = tf.nn.sigmoid(out) #> 0.5
# likelihood = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(x, [-1, 28, 28, 1]), logits=out))
with tf.name_scope("likelihood"):
    likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=out),1)
    likelihood_ = tf.reduce_mean(likelihood)
    variable_summaries(likelihood_)
with tf.name_scope("loss"):
    loss =  tf.reduce_mean(likelihood - tf.reduce_sum(0.5 * (1.+ tf.log(sd ** 2) - mu ** 2 - sd ** 2), 1))
    variable_summaries(loss)

#---------------------------- Updates  ------------------------------

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#---------------------------- Summaries ------------------------------

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

#-------------------------------------- Clearning GPU space after script is done -----------------------------------

gpu_options = tf.GPUOptions()
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#---------------------------- Training ------------------------------
for _ in range(max_epochs):
    np.random.shuffle(train)
    for i in range(train.shape[0]/batch_size):
        batch = train[i*batch_size : (i+1)*batch_size, :]
        tr_lik, _ = sess.run([likelihood_, train_step], 
                             feed_dict={x:batch, phase_train:True, z:np.ones((100,hidden_dim)),gen_image:False})
        if i%50==0:
            summary = sess.run([merged],feed_dict = {x:batch, phase_train:True, z:np.ones((100,hidden_dim)),gen_image:False})
            train_writer.add_summary(summary, i)
        
    test_lik = sess.run([likelihood_],
                        feed_dict={x:test[:100], phase_train:False, z:np.ones((100,hidden_dim)),gen_image:False})
    summary = sess.run([merged],feed_dict = {x:test[:100], phase_train:False, z:np.ones((100,hidden_dim)),gen_image:False})
    test_writer.add_summary(summary, _)
    print "test : " + str(test_lik)
    

    # val_lik = sess.run([likelihood_],
    #                    feed_dict={x:valid[:100], phase_train:False, z:np.ones((100,hidden_dim)),gen_image:False})
    # print "validation : " + str(val_lik)
