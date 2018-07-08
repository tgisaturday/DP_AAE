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
            output = tf.nn.relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        W_fc1 = tf.Variable(tf.random_normal([4*4*512, 100]))
        theta_G.append(W_fc1)
        z = tf.matmul(tf.layers.flatten(current_input),W_fc1)
        z =  tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z = tf.nn.tanh(z)
        z_value = z
        W_fc2 = tf.Variable(tf.random_normal([100, 4*4*512]))
        theta_G.append(W_fc2)
        z_ = tf.matmul(z,W_fc2)
        z_ = tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, 512])
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
            output = tf.nn.relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse()
        z = tf.matmul(tf.layers.flatten(current_input), tf.transpose(W_fc2))
        z = tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z = tf.nn.tanh(z)
        z_value = z
        z_ = tf.matmul(z, tf.transpose(W_fc1))
        z_ =  tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, 512])
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



W1 = tf.Variable(xavier_init([3,3,3,64]))
W2 = tf.Variable(xavier_init([3,3,64,64]))
W3 = tf.Variable(xavier_init([3,3,64,128]))
W4 = tf.Variable(xavier_init([3,3,128,128]))
W5 = tf.Variable(xavier_init([3,3,128,256]))
W6 = tf.Variable(xavier_init([3,3,256,256]))
W7 = tf.Variable(xavier_init([4096, 1]))
b7 = tf.Variable(tf.zeros(shape=[1]))
W_fc = tf.Variable(xavier_init([4096,4096]))
W_z = tf.Variable(xavier_init([4096, 100]))
b_z=tf.Variable(tf.zeros([100]))
theta_D = [W1,W2,W3,W4,W5,W6,W7,b7,W_fc, W_z,b_z]

def discriminator(x):
    if len(x.get_shape()) == 3:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, 32, 32, 3])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')   
    with tf.name_scope("Discriminator"):
        conv1 = tf.nn.conv2d(x_tensor, W1, strides=[1,1,1,1],padding='SAME')
        conv1 = tf.contrib.layers.batch_norm(conv1,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h1 = tf.nn.leaky_relu(conv1)
    
        conv2 = tf.nn.conv2d(h1, W2, strides=[1,2,2,1],padding='SAME')
        conv2 = tf.contrib.layers.batch_norm(conv2,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h2 = tf.nn.leaky_relu(conv2)
    

        conv3 = tf.nn.conv2d(h2, W3, strides=[1,1,1,1],padding='SAME')
        conv3 = tf.contrib.layers.batch_norm(conv3,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h3 = tf.nn.leaky_relu(conv3)
        

        conv4 = tf.nn.conv2d(h3, W4, strides=[1,2,2,1],padding='SAME')
        conv4 = tf.contrib.layers.batch_norm(conv4,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h4 = tf.nn.leaky_relu(conv4)

        conv5 = tf.nn.conv2d(h4, W5, strides=[1,1,1,1],padding='SAME')
        conv5 = tf.contrib.layers.batch_norm(conv5,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h5 = tf.nn.leaky_relu(conv5)
        
        conv6 = tf.nn.conv2d(h5, W6, strides=[1,2,2,1],padding='SAME')
        conv6 = tf.contrib.layers.batch_norm(conv6,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        h6 = tf.nn.leaky_relu(conv6)

        h7 = tf.layers.flatten(h6)
     
        d = tf.nn.xw_plus_b(h7, W7, b7)
        
        z_value = tf.matmul(h7,W_fc)
        z_value = tf.contrib.layers.batch_norm(z_value,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_value = tf.nn.leaky_relu(z_value)
        z_value = tf.nn.xw_plus_b(z_value,W_z,b_z)
        z_value = tf.nn.tanh(z_value)
    return d, z_value

G_logits,G_sample,A_logits,A_sample, gen_real_z = autoencoder(X)

D_real_logits, disc_real_z = discriminator(X)
D_fake_logits, disc_fake_z = discriminator(G_sample)
#A_fake_logits, _ = discriminator(A_sample)
A_true_flat = tf.reshape(X, [-1,32,32,3])

global_step = tf.Variable(0, name="global_step", trainable=False)
A_loss = tf.reduce_mean(tf.pow(A_true_flat - A_sample, 2))
D_z_loss =tf.reduce_mean(tf.pow(disc_fake_z - gen_real_z, 2))
D_loss = tf.reduce_mean(D_fake_logits)-tf.reduce_mean(D_real_logits) + 100.0*D_z_loss
G_loss = -tf.reduce_mean(D_fake_logits)- 100.0*D_z_loss + 100.0*A_loss


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
tf.summary.scalar('A_loss',A_loss)
tf.summary.scalar('D_z_loss',D_z_loss)
merged = tf.summary.merge_all()

num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
D_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9)
G_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9)


D_grads_and_vars=D_optimizer.compute_gradients(D_loss, var_list=theta_D)
G_grads_and_vars=G_optimizer.compute_gradients(G_loss, var_list=theta_G)


D_solver = D_optimizer.apply_gradients(D_grads_and_vars, global_step=global_step)
G_solver = G_optimizer.apply_gradients(G_grads_and_vars, global_step=global_step)


clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D] 


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
        _, D_loss_curr,_ = sess.run([D_solver, D_loss, clip_D],feed_dict={X: X_mb})
        X_mb = next_batch(mb_size, x_train)        
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