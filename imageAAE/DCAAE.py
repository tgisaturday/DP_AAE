import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os
import math

initializer = tf.contrib.layers.xavier_initializer()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)

def exponential_lambda_decay(seq_lambda, global_step, decay_steps, decay_rate, staircase=False):
    global_step = float(global_step)
    decay_steps = float(decay_steps)
    decay_rate = float(decay_rate)
    p = global_step / decay_steps
    if staircase:
        p = math.floor(p)    
    return seq_lambda * math.pow(decay_rate, p)

def random_laplace(shape,sensitivity, epsilon):
    rand_uniform = tf.random_uniform(shape,-0.5,0.5,dtype=tf.float32)
    rand_lap= - (sensitivity/epsilon)*tf.multiply(tf.sign(rand_uniform),tf.log(1.0 - 2.0*tf.abs(rand_uniform)))
    return tf.clip_by_norm(tf.clip_by_value(rand_lap, -3.0,3.0),sensitivity)

mb_size = 256
X_dim = 784
z_dim = 10
h_dim = 128
len_x_train = 60000
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)
        
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

initializer = tf.contrib.layers.xavier_initializer()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)

X = tf.placeholder(tf.float32, shape=[None, X_dim])

theta_A = []
theta_G = []

def autoencoder(x):
    input_shape=[None, 784]
    n_filters=[1, 10, 10, 10]
    filter_sizes=[3, 3, 3, 3]
    
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(tf.random_uniform([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output],-1.0 / math.sqrt(n_input),
                                          1.0 / math.sqrt(n_input)))
        theta_A.append(W)
        b = tf.Variable(tf.zeros([n_output]))
        theta_A.append(b)
        encoder.append(W)
        output = tf.nn.leaky_relu(tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # store the latent representation
    z = current_input    
    encoder.reverse()
    shapes.reverse()
    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W_enc = encoder[layer_i]
        b_enc = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        theta_A.append(W_enc)
        theta_A.append(b_enc)
        output = tf.nn.leaky_relu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W_enc,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b_enc))
        current_input = output
        
    y = current_input
    
    enc_noise = random_laplace(shape=tf.shape(z),sensitivity=1.0,epsilon=0.2)
    z = tf.add(z,enc_noise)
    current_infer = z
    
    for layer_i, shape in enumerate(shapes):
        W_enc = encoder[layer_i]
        W = tf.Variable(tf.random_uniform(W_enc.get_shape().as_list(),-1.0 / math.sqrt(n_input),
                                          1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        theta_G.append(W)
        theta_G.append(b)
        output = tf.nn.leaky_relu(tf.add(
            tf.nn.conv2d_transpose(
                current_infer, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_infer = output
    g = current_infer
    return y, g
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

D_W1 = tf.Variable(tf.random_uniform([4,4,1,64],-1.0 ,1.0), name='W1')
D_W2 = tf.Variable(tf.random_uniform([4,4,64,128], -1.0/8,1.0/8), name='W2')
D_fc1 = tf.Variable(xavier_init([7*7*128, 1024]))
D_b1 = tf.Variable(tf.zeros(shape=[1024]))
D_fc2 = tf.Variable(xavier_init([1024,1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2,D_fc1,D_fc2, D_b1, D_b2]

def discriminator(x):
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, 1])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')   
   
    conv1 = tf.nn.conv2d(x_tensor, D_W1, strides=[1,2,2,1],padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1,center=True, scale=True,is_training=True)
    h1 = tf.nn.leaky_relu(conv1, 0.2)
    
    conv2 = tf.nn.conv2d(h1, D_W2, strides=[1,2,2,1],padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2,center=True, scale=True,is_training=True)
    h2 = tf.nn.leaky_relu(conv2, 0.2)
  
    h2 = tf.layers.flatten(h2)

    h2 = tf.matmul(h2, D_fc1) + D_b1
    h2 = tf.contrib.layers.batch_norm(h2,center=True, scale=True,is_training=True)
    h3 = tf.nn.leaky_relu(h2, 0.2)
    
    d =  tf.matmul(h3, D_fc2) + D_b2
    return d


# Prediction
A_sample, G_sample = autoencoder(X)
G_true = X

D_real = discriminator(X)
D_fake = discriminator(G_sample)
reg_loss = tf.nn.l2_loss(D_fc2)
A_true_flat = tf.reshape(X, [-1,28,28,1])

global_step = tf.Variable(0, name="global_step", trainable=False)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake) + tf.reduce_mean(reg_loss)
A_loss = tf.reduce_mean(tf.pow(A_true_flat -A_sample, 2))
G_loss = -tf.reduce_mean(D_fake)+ tf.reduce_mean(reg_loss)
tf.summary.scalar('D_loss',D_loss)
tf.summary.scalar('G_loss',G_loss)
tf.summary.scalar('A_loss',A_loss)

merged = tf.summary.merge_all()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

with tf.control_dependencies(update_ops):
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=theta_D,global_step=global_step)
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G,global_step=global_step)
    A_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(A_loss, var_list=theta_A,global_step=global_step)

if not os.path.exists('dc_out/'):
    os.makedirs('dc_out/')
    
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/home/tgisaturday/Workspace/Taehoon/DP_AAE/imageAAE'+'/graphs/'+'mnist',sess.graph)
    sess.run(tf.global_variables_initializer())
    i = 0
    for it in range(1000000):
        for _ in range(5):
            X_mb, _ = mnist.train.next_batch(mb_size)
            _, D_loss_curr,_ = sess.run([D_solver, D_loss, clip_D],feed_dict={X: X_mb})
            _, A_loss_curr = sess.run([A_solver, A_loss],feed_dict={X: X_mb})
        summary,_, G_loss_curr = sess.run([merged,G_solver, G_loss],feed_dict={X: X_mb})
        current_step = tf.train.global_step(sess, global_step)
        train_writer.add_summary(summary,current_step)
        if it % 100 == 0:
            print('Iter: {}; A_loss: {:.4}; D_loss: {:.4}; G_loss: {:.4};'.format(it,A_loss_curr, D_loss_curr,G_loss_curr))

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={X: X_mb})
            samples_flat = tf.reshape(samples,[-1,784]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
   
        if it% 100000 == 0:
            for ii in range(len_x_train//100):
                xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
                samples = sess.run(G_sample, feed_dict={X: xt_mb})
                if ii == 0:
                    generated = samples
                    labels = y_mb
                else:
                    np.append(generated,samples,axis=0)
                    np.append(labels,y_mb, axis=0)
                    
            np.save('generated_{}_image.npy'.format(str(it)), generated)
            np.save('generated_{}_label.npy'.format(str(it)), samples)

for iii in range(len_x_train//100):
    xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
    samples = sess.run(G_sample, feed_dict={X: xt_mb})
    if iii == 0:
        generated = samples
        labels = y_mb
    else:
        np.append(generated,samples,axis=0)
        np.append(labels,y_mb, axis=0)

np.save('generated_{}_image.npy'.format(str(it)), generated)
np.save('generated_{}_label.npy'.format(str(it)), samples)
                
                