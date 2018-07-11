import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os
import math
import cv2
import time
from scipy.misc import toimage
from utils import add_noise_to_gradients
import scipy.io as sio
initializer = tf.contrib.layers.xavier_initializer()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)

mb_size = 256
X_dim = 1024
len_x_train = 604388

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
        img = sample.reshape(32, 32,3)
        plt.imshow(toimage(img),interpolation='nearest')

    return fig

initializer = tf.contrib.layers.xavier_initializer()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

train_location = '../data/SVHN/train_32x32.mat'
extra_location = '../data/SVHN/extra_32x32.mat'

train_dict = sio.loadmat(train_location)
x_ = np.asarray(train_dict['X'])
x_train = []
for i in range(x_.shape[3]):
    x_train.append(x_[:,:,:,i])
x_train = np.asarray(x_train)

extra_dict = sio.loadmat(extra_location)
x_ex = np.asarray(extra_dict['X'])
x_extra = []
for i in range(x_ex.shape[3]):
    x_extra.append(x_ex[:,:,:,i])
x_extra = np.asarray(x_extra)

x_train = np.concatenate((x_train, x_extra), axis=0)
x_train = normalize(x_train)


theta_A = []
theta_G = []
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def autoencoder(x):
    input_shape=[None, 32, 32, 3]
    n_filters=[3, 128, 256, 512]
    filter_sizes=[5, 5, 5, 5]
    
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
        z = current_input
        z_value = tf.layers.flatten(z)
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            W = tf.Variable(xavier_init(W_enc.get_shape().as_list()))
            theta_G.append(W)
            decoder.append(W)
            shapes_dec.append(current_input.get_shape().as_list())
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            if layer_i == 2:
                output = tf.nn.sigmoid(deconv)
            else:
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
        z = current_input
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            deconv = tf.nn.conv2d_transpose(current_input, W_enc,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            if layer_i == 2:
                output = tf.nn.sigmoid(deconv)
            else:
                output = tf.nn.relu(deconv)
            current_input = output
        a = current_input
        a_logits = deconv        

    return g_logits, g, a_logits, a, z_value


W1_D = tf.Variable(xavier_init([5,5,3,64]))
W2_D = tf.Variable(xavier_init([5,5,64,128]))
W3_D = tf.Variable(xavier_init([5,5,128,256]))
W4_D = tf.Variable(xavier_init([4096, 1]))
b4_D = tf.Variable(tf.zeros(shape=[1]))
theta_D = [W1_D,W2_D,W3_D,W4_D,b4_D]

def discriminator(x):
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(x, [-1, 32, 32, 3])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')   
    with tf.name_scope("Discriminator"):
        conv1 = tf.nn.conv2d(x_tensor, W1_D, strides=[1,2,2,1],padding='SAME')
        conv1 = tf.contrib.layers.batch_norm(conv1,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h1 = tf.nn.leaky_relu(conv1)
    
        conv2 = tf.nn.conv2d(h1, W2_D, strides=[1,2,2,1],padding='SAME')
        conv2 = tf.contrib.layers.batch_norm(conv2,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h2 = tf.nn.leaky_relu(conv2)
    

        conv3 = tf.nn.conv2d(h2, W3_D, strides=[1,2,2,1],padding='SAME')
        conv3 = tf.contrib.layers.batch_norm(conv3,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h3 = tf.nn.leaky_relu(conv3)

        h4 = tf.layers.flatten(h3)
     
        d = tf.nn.xw_plus_b(h4, W4_D, b4_D)
        
    return d
W1_H = tf.Variable(xavier_init([5,5,3,128]))
W2_H = tf.Variable(xavier_init([5,5,128,256]))
W3_H = tf.Variable(xavier_init([5,5,256,512]))

theta_H = [W1_H,W2_H,W3_H]

def hacker(x):
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(x, [-1, 32, 32, 3])
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

        z = tf.layers.flatten(h3)
     
    return z
G_logits,G_sample,A_logits,A_sample, gen_real_z = autoencoder(X)

D_real_logits = discriminator(X)
disc_real_z = hacker(X)
D_fake_logits = discriminator(G_sample)
disc_fake_z = hacker(G_sample)
A_true_flat = tf.reshape(X, [-1,32,32,3])

global_step = tf.Variable(0, name="global_step", trainable=False)
A_loss = tf.reduce_mean(tf.pow(A_true_flat - A_sample, 2))
D_z_loss =tf.reduce_mean(tf.pow(disc_fake_z - gen_real_z, 2))
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,labels=tf.ones_like(D_real_logits)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,labels=tf.zeros_like(D_fake_logits)))
G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,labels=tf.ones_like(D_fake_logits)))
D_loss = D_loss_real+D_loss_fake
H_loss = D_z_loss
G_loss = G_loss_fake - D_z_loss + A_loss

tf.summary.image('Original',A_true_flat)
tf.summary.image('G_sample',G_sample)
tf.summary.image('A_sample',A_sample)
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss',G_loss_fake)
tf.summary.scalar('A_loss',A_loss)
tf.summary.scalar('H_loss',H_loss)
merged = tf.summary.merge_all()

num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
D_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5, beta2=0.9)
G_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5, beta2=0.9)
H_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5, beta2=0.9)

D_grads_and_vars=D_optimizer.compute_gradients(D_loss, var_list=theta_D)
G_grads_and_vars=G_optimizer.compute_gradients(G_loss, var_list=theta_G)
H_grads_and_vars=H_optimizer.compute_gradients(H_loss, var_list=theta_H)

D_solver = D_optimizer.apply_gradients(D_grads_and_vars, global_step=global_step)
G_solver = G_optimizer.apply_gradients(G_grads_and_vars, global_step=global_step)
H_solver = H_optimizer.apply_gradients(H_grads_and_vars, global_step=global_step)
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "models/svhn_" + timestamp))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists('models/'):
    os.makedirs('models/')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())  

if not os.path.exists('dc_out_svhn/'):
    os.makedirs('dc_out_svhn/')

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('graphs/'+'svhn',sess.graph)
    sess.run(tf.global_variables_initializer())
    i = 0   
    
    for it in range(1000000000):
        X_mb = next_batch(mb_size, x_train)
        _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: X_mb})
        summary,_, G_loss_curr,A_loss_curr = sess.run([merged,G_solver, G_loss, A_loss],feed_dict={X: X_mb})
        current_step = tf.train.global_step(sess, global_step)
        train_writer.add_summary(summary,current_step)
        
        if it % 100 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4};  A_loss: {:.4};'.format(it,D_loss_curr, G_loss_curr, A_loss_curr))

        if it % 1000 == 0: 
            samples = sess.run(G_sample, feed_dict={X: X_mb})
            samples_flat = tf.reshape(samples,[-1,32,32,3]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out_svhn/{}_G.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            samples = sess.run(A_sample, feed_dict={X: X_mb})
            samples_flat = tf.reshape(samples,[-1,32,32,3]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out_svhn/{}_A.png'.format(str(i).zfill(3)), bbox_inches='tight')
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
                    
            np.save('./generated_svhn/generated_{}_image.npy'.format(str(it)), generated)
            np.save('./generated_svhn/generated_{}_label.npy'.format(str(it)), labels)

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

np.save('./generated_svhn/generated_{}_image.npy'.format(str(it)), generated)
np.save('./generated_svhn/generated_{}_label.npy'.format(str(it)), labels)
'''             
