# -*- coding: utf-8 -*-
"""
Building a new chatbot with deep NLP
Created on Sat Jul 20 23:55:07 2019

@author: Avantika
"""

import numpy as np
import tensorflow as tf
import re
import time

#now import the dataset
lines = open('movie_lines.txt', encoding='utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors = 'ignore').read().split('\n')


#create python dict to map lines and its ids

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]] = _line[4]
        
#create a list of all the conversations(we have lot of metadata and we want to keep only the meaninginful data)
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")# avoiding repeated brackets and spaces
    conversations_ids.append(_conversation.split(','))
        
#getting seperately questions and answers, but make sure the indexes are aligned
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
#cleaning of the texts        
def clean_text(text):
    #put in lower cases
    text = text.lower()
    #remove apostrophe
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"wont't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

#cleaning question
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#cleaning answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


#creating a dictionary that maps each word to tis number of occurences (freq dict)
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1;
        else:
            word2count[word] +=1;
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1;
        else:
            word2count[word] +=1;  
            
 #creating two dictionaries that map the questions words and the answers words to a unique integer           
threshold = 20 #words whose freq > threshold, are in consideration
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
         
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1 
         
         
         
#adding the last tokens to there dictionaries -> usefule for seq2seq endcoder and decoder
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']         

#now add these to or prev dictionaries
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1
for token in tokens:
    answerswords2int[token] = len(answerswords2int)+1    


#creating the inverse dictionary of the answerswords2int dictionary
answersints2word= {w_i:w for w , w_i in answerswords2int.items()}
    

#adding the <EOS> toke to the end of every clean answer, to be fed in decoding part

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
   
#translating all the questions and the answers into integers
    #replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)        


answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

#sorting questions and answers by the length of the questions
#because this will speed up the training, less loss reduces the amount of padding

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            







#creating placeholders for the inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')            
    return inputs, targets, lr, keep_prob

#preprocessing the targets - because decoder will only accept in special format, starting with SOS token
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


#creating the encoder RNN layer, drop out regularization
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    #now lstm is ready, now we will create encoder->which is composed of rnn cells
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell, sequence_length = sequence_length, inputs = rnn_inputs, dtype = tf.float32)
    return encoder_state


#decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    #get the attention states
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name= 'attn_dec_train')    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

#decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    #get the attention states
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, name= 'attn_dec_test')    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
    





    
        
    
    