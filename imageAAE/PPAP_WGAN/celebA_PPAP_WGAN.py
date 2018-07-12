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

mb_size = 64
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

download_celeb_a("../data")
data_files = glob(os.path.join("../data/celebA/*.jpg"))
len_x_train = len(data_files)
sample = [get_image(sample_file, 108, True, 64, is_grayscale = 0) for sample_file in data_files]
sample_images = np.array(sample).astype(np.float32)  
x_train = sample_images

x_train = normalize(x_train)

theta_A = []
theta_G = []
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def autoencoder(x):
    input_shape=[None, 64, 64, 3]
    n_filters=[3, 128, 256, 512, 1024]
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
    decoder = []
    shapes_enc = []
    shapes_dec = []
    with tf.name_scope("Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes_enc.append(current_input.get_shape().as_list())
            W = tf.Variable(xavier_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
            theta_G.append(W)
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        W_fc1 = tf.Variable(tf.random_normal([4*4*1024, 100]))
        theta_G.append(W_fc1)
        z = tf.matmul(tf.layers.flatten(current_input),W_fc1)
        z =  tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z = tf.nn.tanh(z)
        z_value = z
        W_fc2 = tf.Variable(tf.random_normal([100, 4*4*1024]))
        theta_G.append(W_fc2)
        z_ = tf.matmul(z,W_fc2)
        z_ = tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, 1024])
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            W = tf.Variable(xavier_init(W_enc.get_shape().as_list()))
            theta_G.append(W)
            decoder.append(W)
            shapes_dec.append(current_input.get_shape().as_list())
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i ==3:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)  
                output = tf.nn.relu(deconv)
            current_input = output
        g = current_input
        g_logits = deconv
        
        encoder.reverse()
        shapes_enc.reverse()
        decoder.reverse()
        shapes_dec.reverse()
        
    with tf.name_scope("Decoder"):
        for layer_i, shape in enumerate(shapes_dec):
            W_dec = decoder[layer_i]
            conv = tf.nn.conv2d(current_input, W_dec, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        z = tf.matmul(tf.layers.flatten(current_input), tf.transpose(W_fc2))
        z = tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z = tf.nn.tanh(z)
        z_transpose = z
        z_ = tf.matmul(z, tf.transpose(W_fc1))
        z_ =  tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4,1024])            
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            deconv = tf.nn.conv2d_transpose(current_input, W_enc,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i ==3:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)     
                output = tf.nn.relu(deconv)
            current_input = output
        a = current_input
        a_logits = deconv        

    return g_logits, g, a_logits, a, z_value, z_transpose

W1 = tf.Variable(xavier_init([5,5,3,128]))
W2 = tf.Variable(xavier_init([5,5,128,256]))
W3 = tf.Variable(xavier_init([5,5,256,512]))
W4 = tf.Variable(xavier_init([5,5,512,1024]))
W5 = tf.Variable(xavier_init([4*4*1024, 1]))
b5 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [W1,W2,W3,W4,W5,b5]


def discriminator(x):
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(x, [-1, 64, 64, 3])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')   
    with tf.name_scope("Discriminator"):
        conv1 = tf.nn.conv2d(x_tensor, W1, strides=[1,2,2,1],padding='SAME')
        conv1 = tf.contrib.layers.layer_norm(conv1)
        h1 = tf.nn.leaky_relu(conv1)
    
        conv2 = tf.nn.conv2d(h1, W2, strides=[1,2,2,1],padding='SAME')
        conv2 = tf.contrib.layers.layer_norm(conv2)
        h2 = tf.nn.leaky_relu(conv2)
    

        conv3 = tf.nn.conv2d(h2, W3, strides=[1,2,2,1],padding='SAME')
        conv3 = tf.contrib.layers.layer_norm(conv3)
        h3 = tf.nn.leaky_relu(conv3)

        conv4 = tf.nn.conv2d(h3, W4, strides=[1,2,2,1],padding='SAME')
        conv4 = tf.contrib.layers.layer_norm(conv4)
        h4 = tf.nn.leaky_relu(conv4)

        h5 = tf.layers.flatten(h4)
     
        d = tf.nn.xw_plus_b(h5, W5, b5)

    return d

W1_H = tf.Variable(xavier_init([5,5,3,128]))
W2_H = tf.Variable(xavier_init([5,5,128,256]))
W3_H = tf.Variable(xavier_init([5,5,256,512]))
W4_H = tf.Variable(xavier_init([5,5,512,1024]))
W_fc = tf.Variable(xavier_init([4*4*1024, 100]))
theta_H = [W1_H,W2_H,W3_H,W4_H,W_fc]


def hacker(x):
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(x, [-1, 64, 64, 3])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')   
    with tf.name_scope("Hacker"):
        conv1 = tf.nn.conv2d(x_tensor, W1_H, strides=[1,2,2,1],padding='SAME')
        conv1 = tf.contrib.layers.batch_norm(conv1,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h1 = tf.nn.leaky_relu(conv1)
    
        conv2 = tf.nn.conv2d(h1, W2_H, strides=[1,2,2,1],padding='SAME')
        conv2 = tf.contrib.layers.batch_norm(conv2,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h2 = tf.nn.leaky_relu(conv2)
    

        conv3 = tf.nn.conv2d(h2, W3_H, strides=[1,2,2,1],padding='SAME')
        conv3 = tf.contrib.layers.batch_norm(conv3,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h3 = tf.nn.leaky_relu(conv3)
        
        conv4 = tf.nn.conv2d(h3, W4_H, strides=[1,2,2,1],padding='SAME')
        conv4 = tf.contrib.layers.batch_norm(conv4,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h4 = tf.nn.leaky_relu(conv4)
        
        z_value = tf.matmul(tf.layers.flatten(h4),W_fc)
        z_value = tf.nn.tanh(z_value)
    return z_value

G_logits,G_sample,A_logits,A_sample, gen_real_z, gen_trans_z = autoencoder(X)
D_real_logits = discriminator(X)
D_fake_logits = discriminator(G_sample)
#A_fake_logits = discriminator(A_sample)
disc_fake_z = hacker(G_sample)
A_true_flat = tf.reshape(X, [-1,64,64,3])


global_step = tf.Variable(0, name="global_step", trainable=False)
A_loss = tf.reduce_mean(tf.pow(A_true_flat - A_sample, 2))
G_z_loss = tf.reduce_mean(tf.pow(gen_trans_z - gen_real_z, 2))
D_z_loss =tf.reduce_mean(tf.pow(disc_fake_z - gen_real_z, 2))
G_loss = -tf.reduce_mean(D_fake_logits) - 10.0*D_z_loss + 10.0*G_z_loss + 10.0*A_loss
#G_loss = -(0.5*tf.reduce_mean(D_fake_logits) + 0.5*tf.reduce_mean(A_fake_logits)) - 10.0*D_z_loss + 10.0*G_z_loss + 10.0*A_loss
H_loss = 10.0*D_z_loss

D_G_loss = tf.reduce_mean(D_fake_logits)-tf.reduce_mean(D_real_logits)
# Gradient Penalty
epsilon_G = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
X_G_hat = A_true_flat + epsilon_G * (G_sample - A_true_flat)
D_G_X_hat = discriminator(X_G_hat)
grad_D_G_X_hat = tf.gradients(D_G_X_hat, [X_G_hat])[0]
red_G_idx = list(range(1, X_G_hat.shape.ndims))
slopes_G = tf.sqrt(tf.reduce_sum(tf.square(grad_D_G_X_hat), reduction_indices=red_G_idx))
gradient_penalty_G = tf.reduce_mean(tf.square(slopes_G - 1.))
D_G_loss = D_G_loss + 10.0 * gradient_penalty_G

#D_A_loss = tf.reduce_mean(A_fake_logits)-tf.reduce_mean(D_real_logits)
# Gradient Penalty
#epsilon_A = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
#X_A_hat = A_true_flat + epsilon_A * (A_sample - A_true_flat)
#D_A_X_hat = discriminator(X_A_hat)
#grad_D_A_X_hat = tf.gradients(D_A_X_hat, [X_A_hat])[0]
#red_A_idx = list(range(1, X_A_hat.shape.ndims))
#slopes_A = tf.sqrt(tf.reduce_sum(tf.square(grad_D_A_X_hat), reduction_indices=red_A_idx))
#gradient_penalty_A = tf.reduce_mean(tf.square(slopes_A - 1.))
#D_A_loss = D_A_loss + 10.0 * gradient_penalty_A

#D_loss = 0.5*D_G_loss + 0.5*D_A_loss
D_loss = D_G_loss
tf.summary.image('Original',A_true_flat)
tf.summary.image('G_sample',G_sample)
tf.summary.image('A_sample',A_sample)
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss',-tf.reduce_mean(D_fake_logits))
tf.summary.scalar('A_loss',A_loss)
tf.summary.scalar('G_z_loss',G_z_loss)
tf.summary.scalar('D_z_loss',D_z_loss)
merged = tf.summary.merge_all()

num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1

D_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(D_loss,var_list=theta_D, global_step=global_step)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_loss,var_list=theta_G, global_step=global_step)
H_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(H_loss,var_list=theta_H, global_step=global_step)

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
    train_writer = tf.summary.FileWriter('graphs/'+'celebA',sess.graph)
    sess.run(tf.global_variables_initializer())
    i=0    
 
    for it in range(1000000000):
        for _ in range(5):
            X_mb = next_batch(mb_size, x_train)
            _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: X_mb})
            _, H_loss_curr = sess.run([H_solver, H_loss],feed_dict={X: X_mb})            
        summary,_, G_loss_curr,A_loss_curr = sess.run([merged,G_solver, G_loss, A_loss],feed_dict={X: X_mb})
        current_step = tf.train.global_step(sess, global_step)
        train_writer.add_summary(summary,current_step)
        
        if it % 100 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4};  A_loss: {:.4};'.format(it,D_loss_curr, G_loss_curr, A_loss_curr))

        if it % 1000 == 0: 
            samples = sess.run(G_sample, feed_dict={X: X_mb})
            samples_flat = tf.reshape(samples,[-1,64,64,3]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out_celebA/{}_G.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            samples = sess.run(A_sample, feed_dict={X: X_mb})
            samples_flat = tf.reshape(samples,[-1,64,64,3]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out_celebA/{}_A.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)            
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print('Saved model at {} at step {}'.format(path, current_step))
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
