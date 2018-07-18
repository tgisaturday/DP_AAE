import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os
import math
import time
from utils import add_noise_to_gradients
initializer = tf.contrib.layers.xavier_initializer()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)

mb_size = 256
X_dim = 784
z_dim = 10
h_dim = 128
len_x_train = 60000

mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(32,5))
    gs = gridspec.GridSpec(5,32)
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

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def autoencoder(input_shape, n_filters, filter_sizes, x, theta_G):
    current_input = x
    
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
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            W = tf.Variable(xavier_init(W_enc.get_shape().as_list()))
            theta_G.append(W)
            decoder.append(W)
            shapes_dec.append(current_input.get_shape().as_list())
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i == 2:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
                output = tf.nn.relu(deconv)
            current_input = output
        g = current_input
        
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
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            deconv = tf.nn.conv2d_transpose(current_input, W_enc,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')
            if layer_i == 2:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
                output = tf.nn.relu(deconv)
            current_input = output
        a = current_input      

    return g, a

def discriminator(x,var_D):
    current_input = x
    with tf.name_scope("Discriminator"):
        for i in range(len(var_D)-2):
            conv = tf.nn.conv2d(current_input, var_D[i], strides=[1,2,2,1],padding='SAME')
            conv = tf.contrib.layers.layer_norm(conv)
            current_input = tf.nn.leaky_relu(conv)            
        h = tf.layers.flatten(current_input)     
        d = tf.nn.xw_plus_b(h, var_D[-2], var_D[-1])        
    return d
def gradient_penalty(G_sample, A_true_flat, var_D):
    epsilon = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
    X_hat = A_true_flat + epsilon * (G_sample - A_true_flat)
    D_X_hat = discriminator(X_hat,var_D)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    red_idx = list(range(1, X_hat.shape.ndims))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    return gradient_penalty
    
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        #input placeholder
        X = tf.placeholder(tf.float32, shape=[None, X_dim])
        A_true_flat = tf.reshape(X, [-1,28,28,1]) 
        
        #autoencoder variables
        var_G = []
        var_H = []
        input_shape=[None, 28, 28, 1]
        n_filters=[1, 128, 256, 512]
        filter_sizes=[5, 5, 5, 5]
        
        #discriminator variables
        W1 = tf.Variable(xavier_init([5,5,1,128]))
        W2 = tf.Variable(xavier_init([5,5,128,256]))
        W3 = tf.Variable(xavier_init([5,5,256,512]))
        W4 = tf.Variable(xavier_init([4*4*512, 1]))
        b4 = tf.Variable(tf.zeros(shape=[1]))
        var_D = [W1,W2,W3,W4,b4]        
        
        G_G_sample,A_G_sample = autoencoder(input_shape, n_filters, filter_sizes, A_true_flat, var_G)
        G_H_sample, A_H_sample = autoencoder(input_shape, n_filters, filter_sizes, G_G_sample, var_H) 

        D_real_logits = discriminator(A_true_flat, var_D)
        D_G_G_fake_logits = discriminator(G_G_sample, var_D)
        D_G_H_fake_logits = discriminator(G_H_sample, var_D)
        D_A_G_fake_logits = discriminator(A_G_sample, var_D)
        D_H_G_fake_logits = discriminator(A_H_sample, var_D)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        A_G_loss = tf.reduce_mean(tf.pow(A_true_flat - A_G_sample, 2))
        A_H_loss = tf.reduce_mean(tf.pow(G_G_sample - A_H_sample, 2))

        G_loss = -tf.reduce_mean(D_G_G_fake_logits) - 10.0*tf.reduce_mean(tf.pow(A_true_flat - G_H_sample, 2)) + 10.0*A_G_loss
        H_loss = -tf.reduce_mean(D_G_H_fake_logits) + 10.0*tf.reduce_mean(tf.pow(A_true_flat - G_H_sample, 2)) + 10.0*A_G_loss
        
        D_fake_logits = 0.25*(tf.reduce_mean(D_G_G_fake_logits)+
                              tf.reduce_mean(D_G_H_fake_logits)+
                              tf.reduce_mean(D_A_G_fake_logits)+
                              tf.reduce_mean(D_H_G_fake_logits))
        
        gp = 0.25*(gradient_penalty(G_G_sample,A_true_flat, var_D)+
                   gradient_penalty(G_H_sample,A_true_flat, var_D)+
                   gradient_penalty(A_G_sample,A_true_flat, var_D)+
                   gradient_penalty(A_H_sample,A_true_flat, var_D))  
        
        D_loss = D_fake_logits - tf.reduce_mean(D_real_logits)+ 10.0*gp

        tf.summary.image('Original',A_true_flat)
        tf.summary.image('G_encoded',G_G_sample)
        tf.summary.image('G_reconstructed',A_G_sample)
        tf.summary.image('H_decoded',G_H_sample)
        tf.summary.image('H_reconstructed',A_H_sample)
        tf.summary.scalar('D_loss', -D_loss)
        tf.summary.scalar('G_loss',tf.reduce_mean(D_G_G_fake_logits))
        tf.summary.scalar('H_loss',tf.reduce_mean(D_G_H_fake_logits))
        tf.summary.scalar('decoder_loss',tf.reduce_mean(tf.pow(A_true_flat - G_H_sample, 2)))
        tf.summary.scalar('A_G_loss',A_G_loss)
        tf.summary.scalar('A_H_loss',A_H_loss)
        merged = tf.summary.merge_all()

        num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1

        D_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(D_loss,var_list=var_D, global_step=global_step)
        G_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_loss,var_list=var_G, global_step=global_step)
        H_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(H_loss,var_list=var_H, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "models/mnist" + timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())
        if not os.path.exists('dc_out_mnist/'):
            os.makedirs('dc_out_mnist/')
        #if not os.path.exists('generated_mnist/'):
        #    os.makedirs('generated_mnist/')            

        train_writer = tf.summary.FileWriter('graphs/'+'mnist',sess.graph)
        sess.run(tf.global_variables_initializer())
        i = 0       
        for it in range(1000000000):
            for _ in range(5):
                X_mb, Y_mb = mnist.train.next_batch(mb_size)
                _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: X_mb})
                _, H_loss_curr = sess.run([H_solver, H_loss],feed_dict={X: X_mb})
            summary, _, G_loss_curr= sess.run([merged,G_solver, G_loss],feed_dict={X: X_mb})
            current_step = tf.train.global_step(sess, global_step)
            train_writer.add_summary(summary,current_step)
        
            if it % 100 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; H_loss: {:.4};'.format(it,D_loss_curr, G_loss_curr, H_loss_curr))

            if it % 1000 == 0: 
                samples = sess.run(G_G_sample, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1,784]).eval()
                img_set = np.append(X_mb[:32], samples_flat[:32], axis=0)
                
                samples = sess.run(A_G_sample, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1,784]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0) 
                
                samples = sess.run(G_H_sample, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1,784]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0)
                
                samples = sess.run(A_H_sample, feed_dict={X: X_mb})
                samples_flat = tf.reshape(samples,[-1,784]).eval() 
                img_set = np.append(img_set, samples_flat[:32], axis=0)
                
                fig = plot(img_set)
                plt.savefig('dc_out_mnist/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                i += 1
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print('Saved model at {} at step {}'.format(path, current_step))
'''
            if it% 100000 == 0 and it != 0:
                for ii in range(len_x_train//100):
                    xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
                    samples = sess.run(G_sample, feed_dict={X: xt_mb})
                    if ii == 0:
                        generated = samples
                    else:
                        np.append(generated,samples,axis=0)
                np.save('./generated_mnist/generated_{}_image.npy'.format(str(it)), generated)

    for iii in range(len_x_train//100):
        xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
        samples = sess.run(G_sample, feed_dict={X: xt_mb})
        if iii == 0:
            generated = samples
        else:
            np.append(generated,samples,axis=0)
    np.save('./generated_mnist/generated_{}_image.npy'.format(str(it)), generated)
'''            
