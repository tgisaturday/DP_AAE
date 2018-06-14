import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
import math
from model import seq2CNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

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

def exponential_lambda_decay(seq_lambda, global_step, decay_steps, decay_rate, staircase=False):
    global_step = float(global_step)
    decay_steps = float(decay_steps)
    decay_rate = float(decay_rate)
    p = global_step / decay_steps

    if staircase:
        p = math.floor(p)
    
    return seq_lambda * math.pow(decay_rate, p)
def add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale, sparse_grads=False):
    if not isinstance(grads_and_vars, list):
        raise ValueError('`grads_and_vars` must be a list.')

    gradients, variables = zip(*grads_and_vars)
    noisy_gradients = []
    for gradient in gradients:
        if gradient is None:
            noisy_gradients.append(None)
            continue
        if isinstance(gradient, tf.IndexedSlices):
            gradient_shape = gradient.dense_shape
        else:
            gradient_shape = gradient.get_shape()
        noise = tf.truncated_normal(gradient_shape) * gradient_noise_scale
        noisy_gradients.append(gradient + noise)

    return noisy_gradients

def train_cnn(dataset_name):
    """Step 0: load sentences, labels, and training parameters"""
    dataset = './dataset/'+dataset_name+'_csv/train.csv'
    parameter_file = "./parameters.json"
    params = json.loads(open(parameter_file).read())
    max_document_length = params['max_length']
    filter_sizes = list(int(x) for x in params['filter_sizes'].split(','))
  
    if not os.path.exists('out/'):
        os.makedirs('out/')
    x_raw, y_raw, target_raw = data_helper.load_data(dataset,dataset_name, max_document_length)
    word_counts = {}
    count_words(word_counts, x_raw)        
    print("Size of Vocabulary: {}".format(len(word_counts)))

    #vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,min_frequency=params['min_frequency'])
    #vocab_processor.fit_transform(x_raw)
    #vocab_to_int = vocab_processor.vocabulary_._mapping
    
    # Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
    # (https://github.com/commonsense/conceptnet-numberbatch)
    embeddings_index = {}
    with open('dataset/embeddings/numberbatch-en.txt', encoding='utf-8') as f: 
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    print('Word embeddings:', len(embeddings_index))
    # Find the number of words that are missing from CN, and are used more than our threshold.
    missing_words = 0
    threshold = params['min_frequency']
    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1
                missing_ratio = round(missing_words/len(word_counts),4)*100
    print("Number of words missing from CN:", missing_words)
    print("Percent of words that are missing from vocabulary: {0:.2f}%".format(missing_ratio))

    vocab_to_int = {}
    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1

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
    

    embedding_dim = 300
    vocab_size = len(vocab_to_int)
    word_embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32) 
    for word, i in vocab_to_int.items():
        if word in embeddings_index:      
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embeddings_index[word] = new_embedding    
            word_embedding_matrix[i] = new_embeddingword_count = 0
    word_count= 0
    unk_count = 0
    int_summaries, word_count, unk_count = convert_to_ints(target_raw,vocab_to_int, word_count, unk_count)
    int_texts, word_count, unk_count = convert_to_ints(x_raw,vocab_to_int, word_count, unk_count, eos=True) 
    unk_percent = round(unk_count/word_count,4)*100

    print("Total number of words in texts: {}".format(word_count))
    print("Total number of UNKs in  texts: {}".format(unk_count))
    print("Percent of words that are UNK: {0:.2f}%".format(unk_percent))
    
    #Step 1: pad each sentence to the same length and map each word to an id

    x_int = pad_sentence_batch(vocab_to_int,int_texts)
    target_int = pad_sentence_batch(vocab_to_int,int_summaries)
    
    x = np.array(x_int)
    y = np.array(y_raw)
    target = np.array(target_int) 
    t = np.array(list(len(x) for x in x_int))
    s = np.array(list(len(x) for x in x_int))



    #Step 2: shuffle the train set and split the train set into train and dev sets
    shuffle_indices = np.random.permutation(np.arange(len(t)))
    x_train = x[shuffle_indices]
    y_train = y[shuffle_indices]
    target_train = target[shuffle_indices]
    t_train= t[shuffle_indices]
    s_train = s[shuffle_indices]

    print('x_train: {}'.format(len(x_train)))

    #Step 3: build a graph and cnn object
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = seq2CNN(
                embeddings=word_embedding_matrix,
                filter_sizes=filter_sizes,
                max_summary_length=max_document_length,
                rnn_size=params['rnn_size'],
                vocab_to_int = vocab_to_int,
                num_filters=params['num_filters'],
                vocab_size=len(vocab_to_int),
                embedding_size=params['embedding_dim'],
                sensitivity=params['sensitivity'],
                noise_epsilon=params['noise_epsilon'],
                l2_reg_lambda=params['l2_reg_lambda']
                )
            global_step = tf.Variable(0, name="global_step", trainable=False)            
            num_batches_per_epoch = int((len(x_train)-1)/params['batch_size']) + 1
            epsilon = params['epsilon']
            learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,num_batches_per_epoch, 0.95, staircase=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cnn.D_loss, global_step=global_step)
                train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cnn.G_loss,global_step=global_step)            
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, dataset_name + "_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            #for tensorboard
            train_writer = tf.summary.FileWriter('/home/tgisaturday/Workspace/Taehoon/DP_AAE/seq2seqAAE'+'/graphs/train/'+dataset_name+'_'+timestamp,sess.graph)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def D_train_step(x_batch, target_batch,t_batch,s_batch,seq_lambda):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.targets: target_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn.seq_lambda: seq_lambda,                    
                    cnn.is_training: True}
                summary, _, step, D_loss, G_loss,A_loss = sess.run([cnn.merged, train_D, global_step, cnn.D_loss, cnn.G_loss,cnn.A_loss], feed_dict)
                current_step = tf.train.global_step(sess, global_step)
                train_writer.add_summary(summary,current_step)
                return D_loss, G_loss, A_loss
            
            def G_train_step(x_batch, target_batch,t_batch,s_batch,seq_lambda):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.targets: target_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn.seq_lambda: seq_lambda,
                    cnn.is_training: True}
                summary, _, step, D_loss, G_loss,A_loss = sess.run([cnn.merged, train_G, global_step, cnn.D_loss, cnn.G_loss, cnn.A_loss], feed_dict)
                current_step = tf.train.global_step(sess, global_step)
                train_writer.add_summary(summary,current_step)
                return D_loss, G_loss, A_loss
            
            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch,target_batch, t_batch,s_batch,seq_lambda):
                feed_dict = {
                    cnn.input_x: x_batch, 
                    cnn.targets: target_batch,
                    cnn.text_length: t_batch,
                    cnn.summary_length: s_batch,
                    cnn.batch_size: len(x_batch),
                    cnn.dropout_keep_prob: 1.0,
                    cnn.seq_lambda: seq_lambda,                    
                    cnn.is_training: False}
                summary, step, examples = sess.run([cnn.merged,global_step,cnn.training_logits],feed_dict)
                G_samples = []
                for example in examples:
                    pad = vocab_to_int['PAD']
                    result =  " ".join([int_to_vocab[j] for j in example[1:] if j != pad and int_to_vocab.get(j)!=None])
                    G_samples.append(result)
                return G_samples

            # Save the word_to_id map since predict.py needs it
            #vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))

            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train,target_train,t_train,s_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            #Step 4: train the cnn model with x_train and y_train (batch by batch)\
            for train_batch in train_batches:
                x_train_batch,target_train_batch, t_train_batch,s_train_batch = zip(*train_batch)
                current_step = tf.train.global_step(sess, global_step)
                seq_lambda = exponential_lambda_decay(params['seq_lambda'], current_step,num_batches_per_epoch, 0.95, staircase=True)                 
                D_loss, G_loss, A_loss = D_train_step(x_train_batch,target_train_batch,t_train_batch,s_train_batch,seq_lambda)
 
                D_loss, G_loss, A_loss = G_train_step(x_train_batch,target_train_batch,t_train_batch,s_train_batch,seq_lambda)
                #Step 4.1: evaluate the model with x_dev and y_dev (batch by batch)
                if current_step % 100 == 0:
                    print('step: {} D_loss: {:0.4f} G_loss: {:0.4f} A_loss: {:0.4f}'.format(current_step,D_loss,G_loss,A_loss))
                if current_step % 1000 == 0:
                    G_samples = dev_step(x_train_batch,target_train_batch,t_train_batch,s_train_batch,seq_lambda)
                    Original = []
                    for text in x_train_batch:
                        pad = vocab_to_int['PAD']
                        result =  " ".join([int_to_vocab[j] for j in text if j != pad])
                        Original.append(result)
                    fp = open('out/{}.txt'.format(current_step),'w')
                    for i in range(len(G_samples)):
                        print('[Original]',file=fp)
                        print(Original[i],file=fp)
                        print('[Generated]',file=fp)
                        print(G_samples[i],file=fp)
                    fp.close()
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model at {} at step {}'.format(path, current_step))
                
            #Step 5: predict x_test (batch by batch)\
            test_batches = data_helper.batch_iter(list(zip(x,y,target,t,s)), params['batch_size'],1)
            fp = open('final_result.txt','w')
            fp_applied = open('{}_dp_applied.csv'.format(dataset_name),'w')
            for test_batch in test_batches:
                x_test_batch,y_test_batch,target_test_batch, t_test_batch,s_test_batch = zip(*test_batch)
                G_samples = dev_step(x_test_batch,target_test_batch,t_test_batch,s_test_batch,seq_lambda)
                Original = []
                for text in x_test_batch:
                    pad = vocab_to_int['PAD']
                    result =  " ".join([int_to_vocab[j] for j in text if j != pad and int_to_vocab.get(j)!=None])
                    Original.append(result)
                for i in range(len(G_samples)):
                    print('[Original]',file=fp)
                    print(Original[i],file=fp)
                    print('[Generated]',file=fp)
                    print(G_samples[i],file=fp)
                    print('"{}","{}"'.format(y_test_batch[i],G_samples[i]),file=fp_applied)
            fp.close()
            fp_applied.close()

if __name__ == '__main__':
    # python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
    train_cnn(sys.argv[1])
