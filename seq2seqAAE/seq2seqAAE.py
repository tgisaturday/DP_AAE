import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import learn
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['GO']), ending], 1)

    return dec_input 

def encoding_layer(rnn_size, sequence_length, rnn_inputs, keep_prob):
    '''Create the encoding layer'''
    
    with tf.variable_scope('encoder'):

        #cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
        #cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

        #cell_bw = tf.contrib.rnn.GRUCell(rnn_size)   
        #cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

        cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))   
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

        enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length, dtype=tf.float32)

        enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state
def training_decoding_layer(embeddings, dec_embed_input, summary_length, start_token, end_token, dec_cell, initial_state, output_layer, vocab_size, max_summary_length, batch_size,is_training):
    if is_training == True:
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)
    else:
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
        training_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

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

        #lstm = tf.contrib.rnn.GRUCell(rnn_size)
        #dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)

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
 
def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
                
def pad_sentence_batch(vocab_to_int,sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    result = []
    for sentence in sentence_batch:
        result.append(sentence + [vocab_to_int['PAD']] * (max_sentence - len(sentence)))
    return result

def convert_to_ints(text,vocab_to_int, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["UNK"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["EOS"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count

dataset_name = sys.argv[1]

dataset = './dataset/'+dataset_name+'_csv/test.csv'
x_raw = data_helper.load_data_and_labels(dataset,dataset_name)
word_counts = {}

count_words(word_counts, x_raw)        
print("Size of Vocabulary: {}".format(len(word_counts)))

"""Step 1: pad each sentence to the same length and map each word to an id"""
max_document_length = max([len(x.split(' ')) for x in x_raw])
min_document_length = min([len(x.split(' ')) for x in x_raw])
print('The maximum length of all sentences: {}'.format(max_document_length))
print('The minimum length of all sentences: {}'.format(min_document_length))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit_transform(x_raw)
vocab_to_int = vocab_processor.vocabulary_._mapping
    
# Special tokens that will be added to our vocab
codes = ["UNK","PAD","EOS","GO"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word
usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of words: {}".format(len(word_counts)))
print("Number of words we will use: {}".format(len(vocab_to_int)))
print("Percent of words we will use: {0:.2f}%".format(usage_ratio))
    
# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_texts, word_count, unk_count = convert_to_ints(x_raw,vocab_to_int, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in texts: {}".format(word_count))
print("Total number of UNKs in  texts: {}".format(unk_count))
print("Percent of words that are UNK: {0:.2f}%".format(unk_percent))
    
"""Step 1: pad each sentence to the same length and map each word to an id"""
x_int = pad_sentence_batch(vocab_to_int,int_texts)

x = np.array(x_int)
s = np.array(list(max_document_length for x in x_int))
shuffle_indices = np.random.permutation(np.arange(len(s)))
x_train = x[shuffle_indices]

#models
X=tf.placeholder(tf.int32, [None, None], name='input_x')
S=tf.placeholder(tf.int32, (None,), name='text_length')
batch_size = 256
num_epochs = 100000
vocab_size = len(vocab_to_int)
embedding_size = 200
rnn_size = 128
embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
dropout_keep_prob = 0.5
filter_sizes=[3,4,5]

def autoencoder(x,s):
    enc_embed_input = tf.nn.embedding_lookup(embeddings, x)
    enc_output, enc_state = encoding_layer(rnn_size, s, enc_embed_input,dropout_keep_prob)            
    dec_input = process_encoding_input(x, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    training_logits= decoding_layer(dec_embed_input,
                                    embeddings,
                                    enc_output,
                                    enc_state, 
                                    vocab_size, 
                                    s,
                                    s,
                                    max_document_length,
                                    rnn_size, 
                                    vocab_to_int, 
                                    dropout_keep_prob, 
                                    batch_size,
                                    True)

    generated =tf.argmax(training_logits[0].rnn_output,axis=2,name='rnn_output',output_type=tf.int64)
    generated  = tf.reshape(generated, [batch_size,max_document_length])
    return generated , training_logits
filter_shape = [3, embedding_size, 1, 64]
D_W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
filter_shape = [4, embedding_size, 1, 64]
D_W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
filter_shape = [5, embedding_size, 1, 64]
D_W3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
D_Wfc = tf.Variable(tf.truncated_normal([3*64,1], stddev=0.1))
D_b = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2,D_Wfc, D_b]

def discriminator(x):
    cnn_input = tf.nn.embedding_lookup(embeddings,x)
    cnn_input = tf.expand_dims(cnn_input, -1)
    pooled_outputs = []
    conv = tf.nn.conv2d(cnn_input, D_W1, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    #Apply nonlinearity
    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=True)
    h = tf.nn.leaky_relu(h,0.2)
    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(h, ksize=[1, max_document_length - 3 + 1, 1, 1], strides=[1, 1, 1, 1],
                                       padding='VALID', name='pool')                
    pooled_outputs.append(pooled)
    conv = tf.nn.conv2d(cnn_input, D_W2, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    #Apply nonlinearity
    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=True)
    h = tf.nn.leaky_relu(h,0.2)
    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(h, ksize=[1, max_document_length - 4 + 1, 1, 1], strides=[1, 1, 1, 1],
                                       padding='VALID', name='pool')                
    pooled_outputs.append(pooled)
    conv = tf.nn.conv2d(cnn_input, D_W3, strides=[1, 1, 1, 1], padding='VALID', name='conv')
    #Apply nonlinearity
    h = tf.contrib.layers.batch_norm(conv,center=True, scale=True,is_training=True)
    h = tf.nn.leaky_relu(h,0.2)
    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(h, ksize=[1, max_document_length - 5 + 1, 1, 1], strides=[1, 1, 1, 1],
                                       padding='VALID', name='pool')                
    pooled_outputs.append(pooled)    
    
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, 3*64])

    h_drop = tf.nn.dropout(h_pool_flat,dropout_keep_prob)
    scores = tf.nn.xw_plus_b(h_drop, D_Wfc, D_b, name='scores') 
    return scores

G_sample,training_logits = autoencoder(X,S)
G_true = X
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
masks = tf.sequence_mask(s, max_document_length, dtype=tf.float32, name='masks')
A_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(training_logits[0].rnn_output,G_true,masks))
G_loss = -tf.reduce_mean(D_fake)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

with tf.control_dependencies(update_ops):
    D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss)
    A_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(A_loss)


if not os.path.exists('seq_out/'):
    os.makedirs('seq_out/')
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0

    train_batches = data_helper.batch_iter(list(zip(x_train,s)),batch_size,num_epochs)
    it=0
    for train_batch in train_batches:
        X_mb, S_mb = zip(*train_batch)
        _, D_loss_curr,_ = sess.run([D_solver, D_loss, clip_D],feed_dict={X: X_mb,S:S_mb})
        _, A_loss_curr, samples = sess.run([A_solver, A_loss, G_sample],feed_dict={X: X_mb,S:S_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss],feed_dict={X: X_mb,S:S_mb})

        if it % 100 == 0:
            print('Iter: {}; D_loss: {:.4}; A_loss: {:.4}; G_loss: {:.4};'.format(it, D_loss_curr, A_loss_curr,G_loss_curr))

        if it % 1000 == 0:
            pad = vocab_to_int['PAD']
            fp = open('dc_out/{}.txt'.format(str(i).zfill(3)),'w')
            for ii in range(len(samples)):
                print('<Original>',file=fp)
                original = " ".join([int_to_vocab[j] for j in X_mb[0][ii] if j != pad])
                print(original,file=fp)
                print('<Generated>',file=fp)
                sample_text =  " ".join([int_to_vocab[j] for j in samples[ii] if j != pad])
                print(sample_text,file=fp)
            fp.close()
            i += 1
        it+=1

