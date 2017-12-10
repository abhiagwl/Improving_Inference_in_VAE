import tensorflow as tf
import numpy as np
import datetime
import socket
import os
import argparse

#-------------------------------- Hyper Parameters -----------------------------
batch_size = 100
hidden_dim = 30
learning_rate = 0.0005
max_epochs = 50
k=10
scale_param = 10000
sample_param = 100
#-------------------------------- Loading data -----------------------------

train = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_train.amat').readlines()]).astype('float32')
valid = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_valid.amat').readlines()]).astype('float32')
test = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_test.amat').readlines()]).astype('float32')

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, required=True)
parser.add_argument('-k', '--num_of_transforms', type=int, required=True)
parser.add_argument('-e', '--num_of_epoch', type=int, required=True)
parser.add_argument('-n', '--name', type=str, required=False)
parser.add_argument('-s', '--sparam', type=int, required=True)
args = parser.parse_args()
gpu = args.gpu
max_epochs = args.num_of_epoch
k = args.num_of_transforms
name_dir = args.name
sample_param = args.sparam

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
#-------------------------------- Placeholders -----------------------------
# summaries_dir = "../logs/norm_test"
if name_dir is not None:
    summaries_dir = os.path.join('/extra_data/abhiagwl/pml/logs/norm_tests'+'_'+str(name_dir)+"_"+str(k)+"_"+str(scale_param)+"_"+str(max_epochs), datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
else:
    summaries_dir = os.path.join('/extra_data/abhiagwl/pml/logs/norm_tests'+"_"+str(k)+"_"+str(scale_param)+"_"+str(max_epochs), datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())

x = tf.placeholder(tf.float32, shape=[None, 28*28])
phase_train = tf.placeholder(tf.bool)
t = tf.placeholder(tf.float32)
# z = tf.placeholder(tf.float32 , shape=[None,hidden_dim])
# gen_image = tf.placeholder(tf.bool)
#-------------------------------- Utility functions -----------------------------


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # mean = tf.reduce_mean(var)
        tf.summary.scalar('value',var)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)

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


#-------------------------------- norm layer -----------------------------
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
input_z_nf = samples
with tf.name_scope("norm_flow"):  
    with tf.name_scope("norm_flow_1"):
        z_nf, pz_nf = norm_layer(input_z_nf)
        input_z_nf = z_nf
    sum_p_nf = pz_nf
    for i in range(k):
        with tf.name_scope("norm_flow_"+str(i+2)):
            z_nf, pz_nf = norm_layer(input_z_nf)
            input_z_nf  = z_nf
        sum_p_nf = sum_p_nf + pz_nf
    

#-------------------------------------- Decoder -----------------------------------

with tf.name_scope("decoder"):
    # input_z = tf.where(gen_image,z,samples)
    # input_z = samples
    input_z = z_nf
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

with tf.name_scope("likelihood"):
    likelihood_ = tf.reduce_mean(likelihood)
    variable_summaries(likelihood_)

# loss =  tf.reduce_mean(likelihood - tf.reduce_sum(0.5 * (1.+ tf.log(sd ** 2) - mu ** 2 - sd ** 2), axis=1))
# aplha_t = tf.constant(0.01,dtype= tf.float32)
beta_t = tf.minimum(1.,0.01 + t/scale_param)
# beta_t =1.0
with tf.name_scope("loss"):
    loss =  beta_t*tf.reduce_mean(likelihood) -0.5*tf.reduce_mean(tf.reduce_sum(tf.log(sd),axis =1)) + tf.reduce_mean(tf.reduce_sum(0.5*input_z**2 ,axis=1))-tf.reduce_mean(sum_p_nf)
    variable_summaries(loss)

#-------------------------------------- Updates for batch norm and other layers -----------------------------------

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    
#-------------------------------------- Summary & Clearning GPU space after script is done -----------------------------------

tf.summary.scalar('K',k)
tf.summary.scalar('scale_param',scale_param)
merged = tf.summary.merge_all()


gpu_options = tf.GPUOptions()#per_process_gpu_memory_fraction=0.5)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)

test_writer = tf.summary.FileWriter(summaries_dir + '/test')

val_writer = tf.summary.FileWriter(summaries_dir + '/val')

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
        if iteration%50==0:
            tr_lik, _,summary = sess.run([likelihood_, train_step,merged],feed_dict = {x:batch, phase_train:True, t:iteration})
            train_writer.add_summary(summary, iteration)
        else:
            tr_lik, _ = sess.run([likelihood_, train_step], feed_dict={x:batch, phase_train:True,t:(iteration)})
        
                                                                   #, z:np.zeros((batch_size,hidden_dim), gen_image:False) })
        # print tr_lik
    test_marg = 0.0
    val_marg = 0.0
    test_liki = 0.0
    val_liki = 0.0
    for i in range(sample_param):
        test_lik,test_loss,summary_ = sess.run([likelihood_,loss, merged], feed_dict={x:test, phase_train:False,t:scale_param})
        test_liki+=test_lik
        test_marg+=test_loss
        val_lik,val_loss,summary_ = sess.run([likelihood_,loss,merged], feed_dict={x:valid, phase_train:False,t:scale_param})
        val_liki+=val_lik
        val_marg+=val_loss
    print "validation marginal likelihood : " + str(val_marg/sample_param)
    print "test marginal likelihood: " + str(test_marg/sample_param)
    print "validation likelihood : " + str(val_liki/sample_param)
    print "test likelihood: " + str(test_liki/sample_param)
#     for i in range(sample_param):
#         test_lik,test_loss,summary_ = sess.run([likelihood_,loss, merged], feed_dict={x:test, phase_train:False,t:100000})
#         print "test likelihood: " + str(test_lik)
#         print "test loss: " + str(test_loss)
#         # test_writer.add_summary(summary_,iteration)

#         val_lik,val_loss,summary_ = sess.run([likelihood_,loss,merged], feed_dict={x:valid, phase_train:False,t:100000})
#         print "validation likelihood : " + str(val_lik)
#         print "validation loss: " + str(val_loss)
#         # val_writer.add_summary(summary_,iteration)