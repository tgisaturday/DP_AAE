import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os
import math
from utils import add_noise_to_gradients
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

mb_size = 64
X_dim = 784
z_dim = 10
h_dim = 128
len_x_train = 60000

mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)

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
N = tf.placeholder(tf.float32, shape=[None,4,4,128])
theta_A = []
theta_G = []
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
def autoencoder(x):
    input_shape=[None, 28, 28, 1]
    n_filters=[3, 32, 64, 128]
    filter_sizes=[5, 5, 5, 5]
    
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
    current_input = x_tensor
    
    encoder = []
    shapes = []
    
    with tf.name_scope("Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
            W = tf.Variable(xavier_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
            b = tf.Variable(tf.zeros([n_output]))
            theta_A.append(W)
            theta_A.append(b)
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')
            conv = tf.add(conv,b)            
            conv = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=True)
            output = tf.nn.relu(conv)
            current_input = output
    encoder.reverse()
    shapes.reverse()
    # store the latent representation
    z = current_input 
    with tf.name_scope("Decoder"):
        for layer_i, shape in enumerate(shapes):
            W_enc = encoder[layer_i]
            b = tf.Variable(tf.zeros(W_enc.get_shape().as_list()[2]))
            theta_A.append(b)     
            deconv = tf.nn.conv2d_transpose(current_input, W_enc,
                                         tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                           strides=[1, 2, 2, 1], padding='SAME')
            deconv = tf.add(deconv,b)
            deconv = tf.contrib.layers.batch_norm(deconv,center=True, scale=True,is_training=True)
            if layer_i == 3:
                output = tf.nn.sigmoid(deconv)
            else:
                output = tf.nn.relu(deconv)
            current_input = output
        a = current_input
        a_logits = deconv     
    #current_infer = tf.contrib.layers.batch_norm(tf.add(z,N),center=True, scale=True,is_training=True)
    current_infer = z
    with tf.name_scope("Generator"):
        for layer_i, shape in enumerate(shapes):
            W_enc = encoder[layer_i]
            W = tf.Variable(xavier_init(W_enc.get_shape().as_list()))
            b = tf.Variable(tf.zeros(W_enc.get_shape().as_list()[2]))           
            theta_G.append(W)
            theta_G.append(b)     
            deconv = tf.nn.conv2d_transpose(current_infer, W,
                                         tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                           strides=[1, 2, 2, 1], padding='SAME')
            deconv = tf.add(deconv,b)
            deconv = tf.contrib.layers.batch_norm(deconv,center=True, scale=True,is_training=True)
            if layer_i == 3:
                output = tf.nn.sigmoid(deconv)
            else:
                output = tf.nn.relu(deconv)
            current_infer = output
        g = current_infer
        g_logits = deconv

    return g_logits, g, a_logits, a

W1 = tf.Variable(xavier_init([3,3,1,32]))
b1 = tf.Variable(tf.zeros(shape=[32]))
W2 = tf.Variable(xavier_init([3,3,32,32]))
b2 = tf.Variable(tf.zeros(shape=[32]))
W3 = tf.Variable(xavier_init([3,3,32,64]))
b3 = tf.Variable(tf.zeros(shape=[64]))
W4 = tf.Variable(xavier_init([3,3,64,64]))
b4 = tf.Variable(tf.zeros(shape=[64]))
W5 = tf.Variable(xavier_init([3,3,64,128]))
b5 = tf.Variable(tf.zeros(shape=[128]))
W6 = tf.Variable(xavier_init([3,3,128,128]))
b6 = tf.Variable(tf.zeros(shape=[128]))
W7 = tf.Variable(xavier_init([2048, 1]))
b7 = tf.Variable(tf.zeros(shape=[1]))
 
theta_D = [W1,W2,W3,W4,W5,W6,W7,b1,b2,b3,b4,b5,b6,b7]

def discriminator(x):
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, 28, 28, 1])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')   
    with tf.name_scope("Discriminator"):
        conv1 = tf.nn.conv2d(x_tensor, W1, strides=[1,1,1,1],padding='SAME')
        conv1 = tf.add(conv1,b1)
        conv1 = tf.contrib.layers.batch_norm(conv1,center=True, scale=True,is_training=True)
        h1 = tf.nn.leaky_relu(conv1,0.2)
    
        conv2 = tf.nn.conv2d(h1, W2, strides=[1,2,2,1],padding='SAME')
        conv2 = tf.add(conv2,b2)
        conv2 = tf.contrib.layers.batch_norm(conv2,center=True, scale=True,is_training=True)
        h2 = tf.nn.leaky_relu(conv2,0.2)
    

        conv3 = tf.nn.conv2d(h2, W3, strides=[1,1,1,1],padding='SAME')
        conv3 = tf.add(conv3,b3)
        conv3 = tf.contrib.layers.batch_norm(conv3,center=True, scale=True,is_training=True)
        h3 = tf.nn.leaky_relu(conv3,0.2)
        

        conv4 = tf.nn.conv2d(h3, W4, strides=[1,2,2,1],padding='SAME')
        conv4 = tf.add(conv4,b4)
        conv4 = tf.contrib.layers.batch_norm(conv4,center=True, scale=True,is_training=True)
        h4 = tf.nn.leaky_relu(conv4,0.2)

        conv5 = tf.nn.conv2d(h4, W5, strides=[1,1,1,1],padding='SAME')
        conv5 = tf.add(conv5,b5)
        conv5 = tf.contrib.layers.batch_norm(conv5,center=True, scale=True,is_training=True)
        h5 = tf.nn.leaky_relu(conv5,0.2)
        
        conv6 = tf.nn.conv2d(h5, W6, strides=[1,2,2,1],padding='SAME')
        conv6 = tf.add(conv6,b6)
        conv6 = tf.contrib.layers.batch_norm(conv6,center=True, scale=True,is_training=True)
        h6 = tf.nn.leaky_relu(conv6,0.2)

        h7 = tf.layers.flatten(h6)
     
        d = tf.nn.xw_plus_b(h7, W7, b7)
    return d

# Prediction
G_logits,G_sample,A_logits, A_sample = autoencoder(X)

D_real_logits = discriminator(X)
D_fake_logits = discriminator(G_sample)
A_true_flat = tf.reshape(X, [-1,28,28,1])

global_step = tf.Variable(0, name="global_step", trainable=False)
reg_loss = tf.reduce_mean(tf.pow(A_true_flat - A_sample, 2))
D_loss = tf.reduce_mean(D_fake_logits)-tf.reduce_mean(D_real_logits)
G_loss = -tf.reduce_mean(D_fake_logits)
#G_loss = -tf.reduce_mean(D_fake_logits)
# Gradient Penalty
epsilon = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
X_hat = A_true_flat + epsilon * (G_sample - A_true_flat)
D_X_hat = discriminator(X_hat)
grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
red_idx = list(range(1, X_hat.shape.ndims))
slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
D_loss = D_loss + 10.0 * gradient_penalty

tf.summary.image('Original',A_true_flat)
tf.summary.image('G_sample',G_sample)
tf.summary.image('A_sample',A_sample)
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss',G_loss)
tf.summary.scalar('reg_loss',reg_loss)

merged = tf.summary.merge_all()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
D_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9)
G_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9)
R_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9)
D_grads_and_vars=D_optimizer.compute_gradients(D_loss, var_list=theta_D)
G_grads_and_vars=G_optimizer.compute_gradients(G_loss, var_list=theta_G)
R_grads_and_vars=R_optimizer.compute_gradients(reg_loss, var_list=theta_A)
#D_grad_noised = add_noise_to_gradients(D_grads_and_vars,1.0)
#G_grad_noised = add_noise_to_gradients(G_grads_and_vars,1.0)
#R_grads_and_vars = add_noise_to_gradients(R_grads_and_vars,1.0)
with tf.control_dependencies(update_ops):
    D_solver = D_optimizer.apply_gradients(D_grads_and_vars, global_step=global_step)
    G_solver = G_optimizer.apply_gradients(G_grads_and_vars, global_step=global_step)
    R_solver = R_optimizer.apply_gradients(R_grads_and_vars, global_step=global_step)
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]    
if not os.path.exists('dc_out_mnist/'):
    os.makedirs('dc_out_mnist/')
if not os.path.exists('generated_mnist/'):
    os.makedirs('generated_mnist/')    
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/home/tgisaturday/Workspace/Taehoon/DP_AAE/imageAAE'+'/graphs/'+'mnist',sess.graph)
    sess.run(tf.global_variables_initializer())
    i = 0
    for it in range(1000000):
        X_mb, Y_mb = mnist.train.next_batch(mb_size)
        enc_noise = np.random.uniform(-0.2,0.2,[mb_size,4,4,128]).astype(np.float32)     
        _, D_loss_curr,_ = sess.run([D_solver, D_loss,clip_D],feed_dict={X: X_mb, N: enc_noise})
        X_mb, Y_mb = mnist.train.next_batch(mb_size)
        enc_noise = np.random.uniform(-0.2,0.2,[mb_size,4,4,128]).astype(np.float32)     
        _, reg_loss_curr = sess.run([R_solver, reg_loss],feed_dict={X: X_mb, N: enc_noise})
        for _ in range(5):
            X_mb, Y_mb = mnist.train.next_batch(mb_size)
            enc_noise = np.random.uniform(-0.2,0.2,[mb_size,4,4,128]).astype(np.float32)     
            summary,_, G_loss_curr,reg_loss_curr  = sess.run([merged,G_solver, G_loss,reg_loss],feed_dict={X: X_mb, N: enc_noise})


        
        current_step = tf.train.global_step(sess, global_step)
        train_writer.add_summary(summary,current_step)
        
        if it % 100 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; reg_loss: {:.4}'.format(it,D_loss_curr,G_loss_curr, reg_loss_curr))

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={X: X_mb,N: enc_noise})
            samples_flat = tf.reshape(samples,[-1,784]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out_mnist/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
'''
        if it% 100000 == 0:
            for ii in range(len_x_train//100):
                xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
                enc_noise = np.random.normal(0.0,1.0,[100,2,2,512]).astype(np.float32) 
                samples = sess.run(G_sample, feed_dict={X: xt_mb,N: enc_noise})
                if ii == 0:
                    generated = samples
                    labels = y_mb
                else:
                    np.append(generated,samples,axis=0)
                    np.append(labels,y_mb, axis=0)
                    
            np.save('./generated_mnist/generated_{}_image.npy'.format(str(it)), generated)
            np.save('./generated_mnist/generated_{}_label.npy'.format(str(it)), labels)

for iii in range(len_x_train//100):
    xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
    enc_noise = np.random.normal(0.0,1.0,[100,2,2,512]).astype(np.float32)
    samples = sess.run(G_sample, feed_dict={X: xt_mb,N: enc_noise, weight: seq_lambda})
    if iii == 0:
        generated = samples
        labels = y_mb
    else:
        np.append(generated,samples,axis=0)
        np.append(labels,y_mb, axis=0)

np.save('./generated_mnist/generated_{}_image.npy'.format(str(it)), generated)
np.save('./generated_mnist/generated_{}_label.npy'.format(str(it)), labels)
                
'''             
