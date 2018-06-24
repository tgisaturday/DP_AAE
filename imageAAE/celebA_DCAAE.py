import tensorflow as tf
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os
import math
import cv2
from scipy.misc import toimage
from glob import glob
from random import shuffle
from download import download_celeb_a
from utils import *
from utils import add_noise_to_gradients
import time
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

mb_size = 128
X_dim = 4096


def next_batch(num, data, shuffle=True):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    if shuffle == True:
        np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)

def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

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
        img = sample.reshape(64, 64,3)
        plt.imshow(toimage(img),interpolation='nearest')

    return fig

initializer = tf.contrib.layers.xavier_initializer()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)

X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

download_celeb_a("./data")
data_files = glob(os.path.join("./data/celebA/*.jpg"))
len_x_train = len(data_files)
sample = [get_image(sample_file, 108, True, 64, is_grayscale = 0) for sample_file in data_files]
sample_images = np.array(sample).astype(np.float32)  
x_train = sample_images

x_train = normalize(x_train)

theta_A = []

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def autoencoder(x):
    input_shape=[None, 64, 64, 3]
    n_filters=[3, 64, 128, 256, 512]
    filter_sizes=[5, 5, 5, 5, 5]
    
    if len(x.get_shape()) == 3:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, 3])
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

    W = tf.Variable(tf.random_normal([4*4*512, 100]))
    b = tf.Variable(tf.random_normal([100]))
    theta_A.append(W)
    theta_A.append(b)
    z = tf.nn.tanh(tf.nn.xw_plus_b(tf.layers.flatten(current_input), W, b))
    
    with tf.name_scope("Decoder"):
        W = tf.Variable(tf.random_normal([100, 4*4*512]))
        b = tf.Variable(tf.random_normal([4*4*512]))
        theta_A.append(W)
        theta_A.append(b)
        z_ = tf.nn.tanh(tf.nn.xw_plus_b(z, W, b))
        current_input = tf.reshape(z_, [-1, 4, 4, 512])
        print(len(shapes))
        for layer_i, shape in enumerate(shapes):
            W_enc = encoder[layer_i]
            b = tf.Variable(tf.zeros(W_enc.get_shape().as_list()[2]))
            theta_A.append(b)     
            deconv = tf.nn.conv2d_transpose(current_input, W_enc,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            deconv = tf.add(deconv,b)
            deconv = tf.contrib.layers.batch_norm(deconv,center=True, scale=True,is_training=True)
            if layer_i == 4:
                output = tf.nn.sigmoid(deconv)
            else:
                output = tf.nn.relu(deconv)
            current_input = output
        g = current_input
        g_logits = deconv     


    return g_logits, g


W1 = tf.Variable(xavier_init([3,3,3,64]))
b1 = tf.Variable(tf.zeros(shape=[64]))
W2 = tf.Variable(xavier_init([3,3,64,64]))
b2 = tf.Variable(tf.zeros(shape=[64]))
W3 = tf.Variable(xavier_init([3,3,64,128]))
b3 = tf.Variable(tf.zeros(shape=[128]))
W4 = tf.Variable(xavier_init([3,3,128,128]))
b4 = tf.Variable(tf.zeros(shape=[128]))
W5 = tf.Variable(xavier_init([3,3,128,256]))
b5 = tf.Variable(tf.zeros(shape=[256]))
W6 = tf.Variable(xavier_init([3,3,256,256]))
b6 = tf.Variable(tf.zeros(shape=[256]))
W7 = tf.Variable(xavier_init([3,3,256,512]))
b7 = tf.Variable(tf.zeros(shape=[512]))    
W8 = tf.Variable(xavier_init([3,3,512,512]))
b8 = tf.Variable(tf.zeros(shape=[512]))
W9 = tf.Variable(xavier_init([8192, 1]))
b9 = tf.Variable(tf.zeros(shape=[1]))
     
theta_D = [W1,W2,W3,W4,W5,W6,W7,W8,W9,b1,b2,b3,b4,b5,b6,b7,b8,b9]

def discriminator(x):
    if len(x.get_shape()) == 3:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, 64, 64, 3])
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
        

        conv7 = tf.nn.conv2d(h6, W7, strides=[1,1,1,1],padding='SAME')
        conv7 = tf.add(conv7,b7)
        conv7 = tf.contrib.layers.batch_norm(conv7,center=True, scale=True,is_training=True)
        h7 = tf.nn.leaky_relu(conv7,0.2)
        

        conv8 = tf.nn.conv2d(h7, W8, strides=[1,2,2,1],padding='SAME')
        conv8 = tf.add(conv8,b8)
        conv8 = tf.contrib.layers.batch_norm(conv8,center=True, scale=True,is_training=True)
        h9 = tf.nn.leaky_relu(conv8,0.2)

        h10 = tf.layers.flatten(h9)
        
        d = tf.nn.xw_plus_b(h10, W9, b9)
    return d

logits,G_sample = autoencoder(X)

D_real_logits = discriminator(X)
D_fake_logits = discriminator(G_sample)
A_true_flat = tf.reshape(X, [-1,64,64,3])

global_step = tf.Variable(0, name="global_step", trainable=False)
reg_loss = tf.reduce_mean(tf.pow(A_true_flat - G_sample, 2))
D_loss = tf.reduce_mean(D_fake_logits)-tf.reduce_mean(D_real_logits)
G_loss = -tf.reduce_mean(D_fake_logits)+reg_loss

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
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss',G_loss)
tf.summary.scalar('reg_loss',reg_loss)

merged = tf.summary.merge_all()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
D_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9)
G_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9)

D_grads_and_vars=D_optimizer.compute_gradients(D_loss, var_list=theta_D)
G_grads_and_vars=G_optimizer.compute_gradients(G_loss, var_list=theta_A)

#D_grad_noised = add_noise_to_gradients(D_grads_and_vars,1.0)
#G_grad_noised = add_noise_to_gradients(G_grads_and_vars,1.0)

with tf.control_dependencies(update_ops):
    D_solver = D_optimizer.apply_gradients(D_grads_and_vars, global_step=global_step)
    G_solver = G_optimizer.apply_gradients(G_grads_and_vars, global_step=global_step)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D] 

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "models/celebA_" + timestamp))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists('models/'):
    os.makedirs('models/')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())
if not os.path.exists('dc_out_celebA/'):
    os.makedirs('dc_out_celebA/')
#if not os.path.exists('generated_celebA/'):
    #os.makedirs('generated_celebA/')      
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/home/tgisaturday/Workspace/Taehoon/DP_AAE/imageAAE'+'/graphs/'+'celebA',sess.graph)
    sess.run(tf.global_variables_initializer())
    i=0
    for it in range(1000000):
        X_mb = next_batch(mb_size, x_train) 
        _, D_loss_curr,_ = sess.run([D_solver, D_loss,clip_D],feed_dict={X: X_mb})      
        X_mb = next_batch(mb_size, x_train)
        summary,_, G_loss_curr, reg_loss_curr  = sess.run([merged,G_solver, G_loss,reg_loss],feed_dict={X: X_mb})
  
        current_step = tf.train.global_step(sess, global_step)
        train_writer.add_summary(summary,current_step)

        
        if it % 100 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; reg_loss: {:.4}'.format(it,D_loss_curr,G_loss_curr, reg_loss_curr))

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={X: X_mb})
            samples_flat = tf.reshape(samples,[-1,64,64,3]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out_celebA/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
''' 
        if it% 100000 == 0:
            for ii in range(len_x_train//100):
                xt_mb, y_mb = next_batch(100,x_train, y_train_one_hot.eval(),shuffle=False)
                enc_noise = np.random.normal(0.0,1.0,[100,2,2,512]).astype(np.float32)
                samples = sess.run(G_sample, feed_dict={X: xt_mb,N: enc_noise})
                if ii == 0:
                    generated = samples
                    labels = y_mb
                else:
                    np.append(generated,samples,axis=0)
                    np.append(labels,y_mb, axis=0)
                    
            np.save('./generated_cifar10/generated_{}_image.npy'.format(str(it)), generated)
            np.save('./generated_cifar10/generated_{}_label.npy'.format(str(it)), labels)

for iii in range(len_x_train//100):
    xt_mb, y_mb = next_batch(100,x_train, y_train_one_hot.eval(),shuffle=False)
    enc_noise = np.random.normal(0.0,1.0,[100,2,2,512]).astype(np.float32)
    samples = sess.run(G_sample, feed_dict={X: xt_mb,N: enc_noise})
    if iii == 0:
        generated = samples
        labels = y_mb
    else:
        np.append(generated,samples,axis=0)
        np.append(labels,y_mb, axis=0)

np.save('./generated_cifar10/generated_{}_image.npy'.format(str(it)), generated)
np.save('./generated_cifar10/generated_{}_label.npy'.format(str(it)), labels)
'''             
