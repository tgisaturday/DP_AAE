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
X_dim = 1024
len_x_train = 60000

def next_batch(num, data, labels,shuffle=True):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    if shuffle == True:
        np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

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
Y = tf.placeholder(tf.float32, [None, 10])

(x_train, y_train), (x_test, y_test) = load_data()
x_train = np.concatenate((x_train, x_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)

x_train = normalize(x_train)
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)


theta_A = []
theta_G = []
theta_C = []
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def autoencoder(x):
    input_shape=[None, 32, 32, 3]
    n_filters=[3, 64, 128, 256, 512, 1024]
    filter_sizes=[5, 5, 5, 5, 5, 5]
    
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

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    with tf.name_scope("Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
            W = tf.Variable(tf.random_uniform([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output],
                                              -1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))
            theta_A.append(W)
            theta_C.append(W)
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')
            conv = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=True)
            output = tf.nn.relu(conv)
            current_input = output
        
    with tf.name_scope("Softmax_Classifier"):
        h = tf.layers.flatten(current_input)
        W_c1 = tf.Variable(xavier_init([1024,1024]))
        b_c1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b')
        theta_C.append(W_c1)
        theta_C.append(b_c1)
        fc1 = tf.nn.xw_plus_b(h, W_c1, b_c1, name='scores')
        fc1 = tf.contrib.layers.batch_norm(fc1,center=True, scale=True,is_training=True)
        fc1 = tf.nn.relu(fc1)
        W_c2 = tf.Variable(xavier_init([1024,10]))
        b_c2 = tf.Variable(tf.constant(0.1, shape=[10]), name='b')
        theta_C.append(W_c2)
        theta_C.append(b_c2)
        scores = tf.nn.xw_plus_b(fc1, W_c2, b_c2, name='scores')

    # store the latent representation
    z = current_input    
    
    encoder.reverse()
    shapes.reverse()
    
    with tf.name_scope("Decoder"):
        for layer_i, shape in enumerate(shapes):
            W_enc = encoder[layer_i]
            W = tf.Variable(tf.random_uniform(W_enc.get_shape().as_list(),-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))                
            theta_A.append(W)
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                            tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                            strides=[1, 2, 2, 1], padding='SAME')
            deconv = tf.contrib.layers.batch_norm(deconv,center=True, scale=True,is_training=True)
            if layer_i == 4:
                output = tf.nn.sigmoid(deconv)
            else:
                output = tf.nn.relu(deconv)
            current_input = output
        logits = deconv   
        y = current_input
    
    #enc_noise = random_laplace(shape=tf.shape(z),sensitivity=1.0,epsilon=0.2)
    #enc_noise =tf.random_normal(shape=tf.shape(z), mean=0.0, stddev=1.0, dtype=tf.float32)
    #z = tf.add(z,enc_noise)
    current_infer = z
    with tf.name_scope("Generator"):
        for layer_i, shape in enumerate(shapes):
            W_enc = encoder[layer_i]
            W = tf.Variable(tf.random_uniform(W_enc.get_shape().as_list(),-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))
            theta_G.append(W)
            deconv = tf.nn.conv2d_transpose(current_infer, W,
                                            tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                            strides=[1, 2, 2, 1], padding='SAME')
            deconv = tf.contrib.layers.batch_norm(deconv,center=True, scale=True,is_training=True)
            if layer_i == 4:
                output = tf.nn.tanh(deconv)
            else:
                output = tf.nn.relu(deconv)
            current_infer = output
        g = current_infer

    return y, g, scores, logits


D_W1 = tf.Variable(xavier_init([5,5,3,32]), name='W1')
D_W2 = tf.Variable(xavier_init([5,5,32,64]), name='W2')
D_W3 = tf.Variable(xavier_init([5,5,64,128]), name='W3')
D_W4 = tf.Variable(xavier_init([5,5,128,256]), name='W4')
D_fc1 = tf.Variable(xavier_init([1024, 1024]))
D_b1 = tf.Variable(tf.zeros(shape=[1024]))
D_fc2 = tf.Variable(xavier_init([1024, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2,D_W3,D_W4,D_fc1,D_b1,D_fc2,D_b2]

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
        conv1 = tf.nn.conv2d(x_tensor, D_W1, strides=[1,2,2,1],padding='SAME')
        conv1 = tf.contrib.layers.batch_norm(conv1,center=True, scale=True,is_training=True)
        h1 = tf.nn.leaky_relu(conv1,0.2)
    
        conv2 = tf.nn.conv2d(h1, D_W2, strides=[1,2,2,1],padding='SAME')
        conv2 = tf.contrib.layers.batch_norm(conv2,center=True, scale=True,is_training=True)
        h2 = tf.nn.leaky_relu(conv2, 0.2)
    
        conv3 = tf.nn.conv2d(h2, D_W3, strides=[1,2,2,1],padding='SAME')
        conv3 = tf.contrib.layers.batch_norm(conv3,center=True, scale=True,is_training=True)
        h3 = tf.nn.leaky_relu(conv3, 0.2)
    
        conv4 = tf.nn.conv2d(h3, D_W4, strides=[1,2,2,1],padding='SAME')
        conv4 = tf.contrib.layers.batch_norm(conv4,center=True, scale=True,is_training=True)
        h4 = tf.nn.leaky_relu(conv4, 0.2)
    
        h4 = tf.layers.flatten(h4)

        h4 = tf.matmul(h4, D_fc1) + D_b1
        h4 = tf.contrib.layers.batch_norm(h4,center=True, scale=True,is_training=True)
        h5 = tf.nn.leaky_relu(h4, 0.2)
    
        d =  tf.matmul(h5, D_fc2) + D_b2
        
    return d


# Prediction
A_sample, G_sample, scores, logits = autoencoder(X)

D_real = discriminator(X)
D_fake = discriminator(G_sample)

A_true_flat = tf.reshape(X, [-1,32,32,3])

global_step = tf.Variable(0, name="global_step", trainable=False)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
#A_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=A_true_flat,logits=logits))
A_loss = tf.reduce_mean(tf.pow(A_true_flat - A_sample, 2))
G_loss = -tf.reduce_mean(D_fake)
C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=scores))

tf.summary.scalar('C_loss',C_loss)
tf.summary.scalar('D_loss',D_loss)
tf.summary.scalar('G_loss',G_loss)
tf.summary.scalar('A_loss',A_loss)

merged = tf.summary.merge_all()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]
        
num_batches_per_epoch = int((60000-1)/256) + 1

learning_rate = tf.train.exponential_decay(5e-3, global_step,num_batches_per_epoch, 0.95, staircase=True)
with tf.control_dependencies(update_ops):
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=theta_D,global_step=global_step)
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G,global_step=global_step)
    A_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(A_loss, var_list=theta_A,global_step=global_step)
    C_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(C_loss, var_list=theta_C,global_step=global_step)
    
if not os.path.exists('dc_out_cifar10/'):
    os.makedirs('dc_out_cifar10/')
if not os.path.exists('generated_cifar10/'):
    os.makedirs('generated_cifar10/')    
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('/home/tgisaturday/Workspace/Taehoon/DP_AAE/imageAAE'+'/graphs/'+'cifar10',sess.graph)
    sess.run(tf.global_variables_initializer())
    i = 0
    for it in range(1000000):
        for _ in range(5):
            X_mb, Y_mb = next_batch(mb_size, x_train, y_train_one_hot.eval())
            _, D_loss_curr,_ = sess.run([D_solver, D_loss, clip_D],feed_dict={X: X_mb})
            _, A_loss_curr = sess.run([A_solver, A_loss],feed_dict={X: X_mb,Y: Y_mb})
            _, C_loss_curr = sess.run([C_solver, C_loss],feed_dict={X: X_mb,Y: Y_mb})
        summary,_, G_loss_curr = sess.run([merged,G_solver, G_loss],feed_dict={X: X_mb,Y: Y_mb})
        current_step = tf.train.global_step(sess, global_step)
        train_writer.add_summary(summary,current_step)
        if it % 100 == 0:
            print('Iter: {}; A_loss: {:.4}; C_loss: {:.4}; D_loss: {:.4}; G_loss: {:.4};'.format(it,A_loss_curr,C_loss_curr, D_loss_curr,G_loss_curr))

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={X: X_mb})
            samples_flat = tf.reshape(samples,[-1,32,32,3]).eval()         
            fig = plot(np.append(X_mb[:32], samples_flat[:32], axis=0))
            plt.savefig('dc_out_cifar10/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
   
        if it% 100000 == 0:
            for ii in range(len_x_train//100):
                xt_mb, y_mb = next_batch(100,x_train, y_train_one_hot.eval(),shuffle=False)
                samples = sess.run(G_sample, feed_dict={X: xt_mb})
                if ii == 0:
                    generated = samples
                    labels = y_mb
                else:
                    np.append(generated,samples,axis=0)
                    np.append(labels,y_mb, axis=0)
                    
            np.save('./generated_cifar10/generated_{}_image.npy'.format(str(it)), generated)
            np.save('./generated_cifar10/generated_{}_label.npy'.format(str(it)), samples)

for iii in range(len_x_train//100):
    xt_mb, y_mb = mnist.train.next_batch(100,shuffle=False)
    samples = sess.run(G_sample, feed_dict={X: xt_mb})
    if iii == 0:
        generated = samples
        labels = y_mb
    else:
        np.append(generated,samples,axis=0)
        np.append(labels,y_mb, axis=0)

np.save('./generated_cifar10/generated_{}_image.npy'.format(str(it)), generated)
np.save('./generated_cifar10/generated_{}_label.npy'.format(str(it)), samples)
                
                
