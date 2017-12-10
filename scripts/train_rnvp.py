import tensorflow as tf
import numpy as np
import datetime
import socket
import os
import argparse


delta = 1e-12
batch_size = 100
hidden_dim = 30
num_transforms = 10
learning_rate = 0.0005
max_epochs = 20
scale_param = 10000
sample_param = 50
batch_param = 10
train = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_train.amat').readlines()]).astype('float32')
valid = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_valid.amat').readlines()]).astype('float32')
test = np.array([[int(i) for i in line.split()] for line in open('../data/binarized_mnist_test.amat').readlines()]).astype('float32')

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, required=True)
parser.add_argument('-k', '--num_of_transforms', type=int, required=True)
parser.add_argument('-e', '--num_of_epoch', type=int, required=True)
parser.add_argument('-n', '--name', type=str, required=False)
parser.add_argument('-s', '--sparam', type=int, required=True)
parser.add_argument('-b', '--bparam', type=int, required=True)
args = parser.parse_args()
gpu = args.gpu
max_epochs = args.num_of_epoch
num_transforms = args.num_of_transforms
sample_param = args.sparam
batch_param = args.bparam
name_dir = args.name
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

if name_dir is not None:
    summaries_dir = os.path.join('/extra_data/abhiagwl/pml/logs/vae-rnvp'+'_'+str(name_dir)+"_"+str(num_transforms), datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
else:
    summaries_dir = os.path.join('/extra_data/abhiagwl/pml/logs/vae-rnvp'+"_"+str(num_transforms), datetime.datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        tf.summary.scalar('value',var)

# custom coupled layers
class realNVP:
    def __init__(self, name):
        self.name = name
        self.log_jacobian = None

    # forward pass through 2 real NVP layers
    def get_scale(self,name):
        scale = tf.get_variable(self.name+"_"+name+"_scale", [1], tf.float32, 
                                  tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.contrib.layers.l2_regularizer(5e-5))
        return scale
    def get_std(self, x):
        epsilon = 1e-3
        batch_mean, batch_var = tf.nn.moments(x,[0])
        # batch_std = 1.0/stf.sqrt(batch_var + epsilon)
        return batch_var + epsilon
    
    def forward(self, inp,phase_train):
        with tf.name_scope(self.name):
            dim = inp.shape[1]/2

            # first coupled layer
            intermediate_top = inp[:, :dim]
            
            s1 = tf.layers.dense(intermediate_top, units=dim*2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            s1 = tf.nn.relu(tf.layers.batch_normalization(s1,training = phase_train))
            s1 = tf.layers.dense(s1, units=dim, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            t1 = tf.layers.dense(intermediate_top, units=dim*2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            t1 = tf.nn.relu(tf.layers.batch_normalization(t1,training = phase_train))
            t1 = tf.layers.dense(t1, units=dim, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            intermediate_bot = inp[:, dim:] * tf.exp(s1) + t1

            outr_top = intermediate_bot
            
            s2 = tf.layers.dense(outr_top, units=dim*2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            s2 = tf.nn.relu(tf.layers.batch_normalization(s2,training = phase_train))
            s2 = tf.layers.dense(s2, units=dim, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            t2 = tf.layers.dense(intermediate_top, units=dim*2, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            t2 = tf.nn.relu(tf.layers.batch_normalization(t2,training = phase_train))
            t2 = tf.layers.dense(t2, units=dim, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            outr_bot = intermediate_top * tf.exp(s2) + t2

            self.log_jacobian = tf.reduce_sum(s1, 1) + tf.reduce_sum(s2, 1) #- 0.5 *tf.reduce_sum(tf.log(batch_std))
            return tf.concat([outr_bot,outr_top], 1)


    # backward pass through 2 real NVP layers
    def backward(inp):
        pass

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[None, 28*28])
phase_train = tf.placeholder(tf.bool)
t = tf.placeholder(tf.float32)


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

    dist = tf.contrib.distributions.MultivariateNormalDiag(loc=[0.]*hidden_dim, scale_diag=[1.]*hidden_dim)
    samples = mu + dist.sample(tf.shape(sd)[0]) * sd

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
with tf.name_scope("transformation"):
    rnvp = {}
    for idx in range(num_transforms):
        rnvp[idx] = realNVP("realNVP" + str(idx))
        samples = rnvp[idx].forward(samples,phase_train)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
with tf.name_scope("decoder"):
    flat = tf.layers.dense(samples, units=32 * 7 * 7, activation=None, kernel_initializer = tf.contrib.layers.xavier_initializer())
    flat = tf.nn.relu(tf.layers.batch_normalization(flat, training=phase_train))
    deconv1 = tf.layers.conv2d_transpose(tf.reshape(flat, [-1, 7, 7, 32]), filters=32, kernel_size=[5,5], strides=(2, 2), padding='same', 
                        kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)
    deconv1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=phase_train))

    deconv2 = tf.layers.conv2d_transpose(deconv1, filters=32, kernel_size=[5,5], strides=(2, 2), padding='same', 
                        kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)
    deconv2 = tf.nn.relu(tf.layers.batch_normalization(deconv2, training=phase_train))

    out = tf.layers.conv2d_transpose(deconv2, filters=1, kernel_size=[5,5], strides=(1, 1), padding='same', 
                        kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None)

reconstructed_image = tf.nn.sigmoid(out) > 0.5
log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(x, [-1, 28, 28, 1]), logits=out), [1, 2, 3])

with tf.name_scope("likelihood"):
    likelihood_ = tf.reduce_mean(log_likelihood)
    variable_summaries(likelihood_)

# beta_t = tf.minimum(1.,0.01 + t/scale_param)
beta_t = 1.0
# 0 transforms is a simple VAE
if not num_transforms:
    loss =  tf.reduce_mean(log_likelihood - 0.5 * tf.reduce_sum(1.+ tf.log(sd ** 2) - mu ** 2 - sd ** 2, 1))
else:
    loss = beta_t*log_likelihood + 0.5 * tf.reduce_sum(samples ** 2, 1) - 0.5 * tf.reduce_sum(tf.log(sd + delta), 1)
    for idx in range(num_transforms):
        loss -= rnvp[idx].log_jacobian
    loss = tf.reduce_mean(loss)
with tf.name_scope("loss"):
    variable_summaries(loss)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    
    
tf.summary.scalar('K',num_transforms)
# tf.summary.scalar('scale_param',scale_param)
merged = tf.summary.merge_all()


gpu_options = tf.GPUOptions()#per_process_gpu_memory_fraction=0.5)
gpu_options.allow_growth=True
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)

test_writer = tf.summary.FileWriter(summaries_dir + '/test')

val_writer = tf.summary.FileWriter(summaries_dir + '/val')

# sess = tf.Session()
sess.run(tf.global_variables_initializer())
iteration = 0.0
for _ in range(max_epochs):
    np.random.shuffle(train)
    for i in range(train.shape[0]/batch_size):
        iteration+=1
        batch = train[i*batch_size : (i+1)*batch_size, :]
        if (iteration%50)==0:
            tr_lik, _,summary = sess.run([likelihood_, train_step,merged],feed_dict = {x:batch, phase_train:True,t:iteration})
            train_writer.add_summary(summary, iteration)
        else:
            tr_lik, _ = sess.run([likelihood_, train_step], feed_dict={x:batch, phase_train:True,t:iteration})
    test_marg = 0
    val_marg = 0
    test_liki = 0
    val_liki = 0
    for i in range(sample_param/batch_param):
        test_lik,test_loss,summary_ = sess.run([likelihood_,loss, merged], feed_dict={x:np.vstack([test]*batch_param), phase_train:False,t:scale_param})
        test_liki+=(test_lik*batch_param)/sample_param
        test_marg+=(test_loss*batch_param)/sample_param
        val_lik,val_loss,summary_ = sess.run([likelihood_,loss,merged], feed_dict={x:np.vstack([valid]*batch_param), phase_train:False,t:scale_param})
        val_liki+=(val_lik*batch_param)/sample_param
        val_marg+=(val_loss*batch_param)/sample_param
    print "validation marginal likelihood : " + str(val_marg)
    print "test marginal likelihood: " + str(test_marg)
    print "validation likelihood : " + str(val_liki)
    print "test likelihood: " + str(test_liki)
    