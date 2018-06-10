import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

initializer = tf.contrib.layers.xavier_initializer()
he_normal = tf.keras.initializers.he_normal()
rand_uniform = tf.random_uniform_initializer(-1,1,seed=2)
regularizer = tf.contrib.layers.l2_regularizer(1e-2)

class seq2CNN(object):  
    def __init__(self,embeddings,filter_sizes, max_summary_length, rnn_size, vocab_to_int, num_filters, vocab_size, embedding_size):
        
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')            
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        self.seq_lambda = tf.placeholder(tf.float32, name='seq_lambda') 
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.D_vars = []
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            #embeddings = tf.get_variable(name='embedding_W', shape=[vocab_size, embedding_size],initializer=rand_uniform)
            embeddings=embeddings
            enc_embed_input = tf.nn.embedding_lookup(embeddings, self.input_x)
            embedding_size = embedding_size

        #seq2seq layers
        with tf.name_scope('seq2seq'):
            batch_size = tf.reshape(self.batch_size, [])
            enc_output, enc_state = encoding_layer(rnn_size, self.text_length, enc_embed_input, self.dropout_keep_prob)
            
            dec_input = process_encoding_input(self.targets, vocab_to_int, batch_size)
            dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
            training_logits = decoding_layer(dec_embed_input,
                                                embeddings,
                                                enc_output,
                                                enc_state, 
                                                vocab_size, 
                                                self.text_length,
                                                self.summary_length,
                                                max_summary_length,
                                                rnn_size, 
                                                vocab_to_int, 
                                                self.dropout_keep_prob, 
                                                batch_size,
                                                self.is_training)

            self.training_logits =tf.argmax(training_logits[0].rnn_output,axis=2,name='rnn_output',output_type=tf.int64)
            self.training_logits = tf.reshape(self.training_logits, [batch_size,max_summary_length])
            
        #VGGnet_Bigram
        with tf.variable_scope('textCNN'):
            decoder_output = tf.nn.embedding_lookup(embeddings, self.training_logits)
            decoder_output_expanded = tf.expand_dims(decoder_output, -1)

            cnn_input = tf.contrib.layers.batch_norm(decoder_output_expanded,center=True, scale=True,is_training=self.is_training)
            
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope('conv-maxpool-%s' % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable(name='W', shape=filter_shape,initializer=he_normal)
                    self.D_vars.append(W)
                    conv = tf.nn.conv2d(cnn_input, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    #Apply nonlinearity
                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(h, name='relu')
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(h, ksize=[1, max_summary_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                       padding='VALID', name='pool')                
                    pooled_outputs.append(pooled)
 
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, len(filter_sizes)*num_filters])
            with tf.variable_scope('output'):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
                W = tf.get_variable('W', shape=[len(filter_sizes)*num_filters, 1],
                                    initializer=initializer,regularizer = regularizer)
                self.D_vars.append(W)
                b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.1))
                self.D_vars.append(b)
                self.D_logit_fake = tf.nn.xw_plus_b(h_drop, W, b, name='scores')

                
        with tf.variable_scope('textCNN', reuse=True):
            inference_output = enc_embed_input
            inference_output_expanded = tf.expand_dims(inference_output, -1)
            inference_cnn_input = tf.contrib.layers.batch_norm(inference_output_expanded,center=True, scale=True,is_training=self.is_training)
            
            inference_pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope('conv-maxpool-%s' % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable(name='W', shape=filter_shape,initializer=he_normal)
                    conv = tf.nn.conv2d(cnn_input, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    #Apply nonlinearity
                    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=self.is_training)
                    h = tf.nn.relu(h, name='relu')
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(h, ksize=[1, max_summary_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                       padding='VALID', name='pool')                
                    inference_pooled_outputs.append(pooled)
 
            inference_h_pool = tf.concat(inference_pooled_outputs, 3)
            inference_h_pool_flat = tf.reshape(inference_h_pool, [-1, len(filter_sizes)*num_filters])
            with tf.variable_scope('output'):
                inference_h_drop = tf.nn.dropout(inference_h_pool_flat, self.dropout_keep_prob)
                W = tf.get_variable('W', shape=[len(filter_sizes)*num_filters, 1],
                                    initializer=initializer,regularizer = regularizer)
                b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.1))
                self.D_logit_real = tf.nn.xw_plus_b(inference_h_drop, W, b, name='inference_scores')          
            
        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):            
            masks = tf.sequence_mask(self.summary_length, max_summary_length, dtype=tf.float32, name='masks')
            seq_loss = tf.contrib.seq2seq.sequence_loss(training_logits[0].rnn_output,self.targets,masks)
            D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real)))
            D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake)))
            self.D_loss = D_loss_real + D_loss_fake
            self.G_loss = self.seq_lambda*tf.reduce_mean(seq_loss) + D_loss_fake
            self.A_loss = tf.reduce_mean(seq_loss)
            tf.summary.scalar('D_loss',self.D_loss)
            tf.summary.scalar('G_loss',self.G_loss)
            tf.summary.scalar('A_loss',self.A_loss)

        self.merged = tf.summary.merge_all()    
        
def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['GO']), ending], 1)

    return dec_input 

def encoding_layer(rnn_size, sequence_length, rnn_inputs, keep_prob):
    '''Create the encoding layer'''
    
    with tf.variable_scope('encoder'):
        cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))   
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

        enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length, dtype=tf.float32)

        enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state

def training_decoding_layer(embeddings, dec_embed_input, summary_length, start_token, end_token, dec_cell, initial_state, output_layer, vocab_size, max_summary_length, batch_size,is_training):

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

    training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                        output_time_major=False,
                                                        impute_finished=True,
                                                        maximum_iterations=max_summary_length)
    return training_logits



def decoding_layer(dec_embed_input,embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, is_training):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    with tf.variable_scope('decoder'):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                   attention_mechanism = attn_mech,
                                                   attention_layer_size = rnn_size,
                                                   name='Attention_Wrapper')
    
    initial_state =dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    initial_state = initial_state.clone(cell_state=enc_state[0])

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(embeddings,dec_embed_input,
                                                  summary_length,                                                                                                                                           vocab_to_int['GO'], 
                                                  vocab_to_int['EOS'],
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_summary_length,
                                                  batch_size,
                                                  True)  

    return training_logits


