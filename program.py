#!/usr/bin/env python3

import os

python_files = []
for root, dirs, files in os.walk('./code'):
    for file in files:
        if file.endswith('.py'):
            python_files += [ root + "/" + file ]

num_files = len(python_files)
print(str(num_files) + " files loaded")

import io

blob = ""
for file in python_files:
    with io.open(file, mode="r", encoding="utf-8") as f:
        blob += f.read()

all_words = blob.split()

print("Read all files")

import re

comments = re.compile(r'#.*')

cleaned_text = comments.sub('', blob)

commas = re.compile(r',')
cleaned_text = commas.sub(' , ', cleaned_text)

dots = re.compile(r'\.')
cleaned_text = dots.sub(' . ', cleaned_text)

# Line breaks can also cause trouble if the next line is not indented

cr = re.compile(r'\n')
cleaned_text = cr.sub(' <EOL> ', cleaned_text)

# Now just add some spaces to all kinds of brackets
cleaned_text = re.sub(r'\(', ' ( ', cleaned_text)
cleaned_text = re.sub(r'\[', ' [ ', cleaned_text)
cleaned_text = re.sub(r'\{', ' { ', cleaned_text)
cleaned_text = re.sub(r'\)', ' ) ', cleaned_text)
cleaned_text = re.sub(r'\]', ' ] ', cleaned_text)
cleaned_text = re.sub(r'\}', ' } ', cleaned_text)

# Equal signs are also special. We don't want something like "a=b" clobber our dictionary,
# so we add spaces. Then we undo this transformation on the comparison operator "==" and some special
# assignments
cleaned_text = re.sub(r'=', ' = ', cleaned_text)
cleaned_text = re.sub(r'=  =', '==', cleaned_text)
cleaned_text = re.sub(r'\+ =', '\+=', cleaned_text)
cleaned_text = re.sub(r'- =', '-=', cleaned_text)
cleaned_text = re.sub(r'\* =', '\*=', cleaned_text)
cleaned_text = re.sub(r'/ =', '/=', cleaned_text)

# Let's do the same with all the operators
cleaned_text = re.sub(r'\+', ' \+ ', cleaned_text)
cleaned_text = re.sub(r'-', ' - ', cleaned_text)
cleaned_text = re.sub(r'\*', ' \* ', cleaned_text)
cleaned_text = re.sub(r'/', ' / ', cleaned_text)

cleaned_text = re.sub(r'\+ =', '\+=', cleaned_text)
cleaned_text = re.sub(r'- =', '-=', cleaned_text)
cleaned_text = re.sub(r'\* =', '\*=', cleaned_text)
cleaned_text = re.sub(r'/ =', '/=', cleaned_text)


print("Done cleaning!")

def create_dicts(word_list):
    
    # The cleaned word list may contain tokens that only consist of whitespace,
    # or that are padded with whitespace.
    # Here we delete all whitespace in a token and drop it if nothing remains
    cleaned_word_list = [ w.strip() for w in word_list if w.strip() != '' ]
    
    # Now we add a token that will later represent everything that we think
    # is the name of a variable, constant, or any other kind of identifier
    cleaned_word_list += ['<ID>']
    
    # Now we make all tokens unique. This is necessary to ensure that the values
    # are uninterrupted
    cleaned_word_list = set(cleaned_word_list)
    
    # Now we will create the tuples that consist of the future
    # key value pairs. We have to cast this into a list because
    # the enumerate object won't return anything after its first
    # usage
    word_idx_pairs = list(enumerate(cleaned_word_list))

    # Dictionary comprehension. This is now trivial thanks to our
    # work before
    w2n_dict = { w: i for i, w in word_idx_pairs}
    n2w_dict = { i: w for i, w in word_idx_pairs}

    return w2n_dict, n2w_dict

all_words = cleaned_text.split(" ")

# The word count should be at least a certain fraction of the total
# number of documents
min_fraction = 0.2
min_count = num_files * min_fraction

# Cut off words that are simply too rare
all_words = cleaned_text.split(" ")

from collections import Counter
c = Counter(all_words)

reduced_words = [ w for w in all_words if c[w] >= min_count ]

w2n_dict, n2w_dict = create_dicts(all_words)
reduced_w2n_dict, reduced_n2w_dict = create_dicts(reduced_words)

print("Dictionary size:  " + str(len(w2n_dict)))
print("Dictionary size:  " + str(len(reduced_w2n_dict)))

def sentence_to_num(sent, ldict):
    transformed = []
    for w in sent.split(" "):
        if w in ldict.keys():
            transformed.append(ldict[w])
        else:
            transformed.append(ldict['<ID>'])
    return transformed

def sent_to_samples(sentence, window_size=1):
    
    # ls stands for labeled sentence (meaning that the
    # end of the sentence is marked with an <EOL>)
    ls = sentence + [ reduced_w2n_dict['<EOL>'] ]
    
    x_vals = [ ls[i:i+window_size] for i in range(len(ls) - window_size) ]
    y_vals = [ ls[i+window_size] for i in range(len(ls) - window_size) ]
        
    return x_vals, y_vals

def make_training_data(cleaned_text, w2n, sampling_rate = 1.0, window_size=1):
    
    lines = [ sentence_to_num(s, w2n) for s in cleaned_text.split("<EOL>") ]
    
    inputs = []
    labels = []

    for line in lines:
        x, y = sent_to_samples(line, window_size=window_size)
        inputs += x
        labels += y
        
    training_inputs = []
    training_labels = []

    for idx in range(len(inputs)):
        if random.random() < sampling_rate:
            training_inputs.append(inputs[idx])
            training_labels.append(labels[idx])
            
    training_inputs = np.array(training_inputs)
    training_labels = to_categorical(training_labels)
    
    return training_inputs, training_labels

import random
import numpy as np
from keras.utils import to_categorical

training_inputs, training_labels = make_training_data(cleaned_text, reduced_w2n_dict)

from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.optimizers import Adam

learning_rate = 1e-4
embedding_dim = 128
mem_size = 128 # Number of recurrent cells (GRU in our case)

epochs = 4
batch_size = 256
validation_split = 0.1

# ----
load_pretrained = True
pretrained_date = "2018-02-20--22-21-59"

# --- Auto computed
vocab_size = len(reduced_w2n_dict)

runs = 1

for mem_size in [ 16, 32, 64, 128]:
	for embedding_dim in [ 16, 32, 64, 128 ]:
		for i in range(runs):

			print("Params: M" + str(mem_size) + " E" + str(embedding_dim))

			model = Sequential()

			model.add( Embedding(vocab_size, embedding_dim, input_length=1) )
			model.add( GRU(mem_size, activation='tanh') )
			model.add( Dense(vocab_size, activation='softmax') )

			model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy')

			# model.summary()

			hist = model.fit(x = training_inputs, y = training_labels, epochs = epochs, \
			  batch_size = batch_size, verbose = 1, validation_split = validation_split)

			with open("gru.txt", "a") as f:
				f.write(str(mem_size) + " " + str(embedding_dim) + " " + str(hist.history['val_loss'][0]))
				f.write("\n")

try:
    del training_inputs
except:
    pass
    
try:
    del training_labels
except:
    pass
window_size = 2
training_inputs, training_labels = make_training_data(cleaned_text, reduced_w2n_dict, window_size = window_size)

for mem_size in [ 16, 32, 64, 128  ]:
	for embedding_dim in [ 16, 32, 64, 128 ]:
		for i in range(runs):

			print("Params: M" + str(mem_size) + " E" + str(embedding_dim))

			model = Sequential()

			model.add( Embedding(vocab_size, embedding_dim, input_length=window_size) )
			model.add( GRU(mem_size, activation='tanh') )
			model.add( Dense(vocab_size, activation='softmax') )

			model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy')

			# model.summary()

			hist = model.fit(x = training_inputs, y = training_labels, epochs = epochs, \
			  batch_size = batch_size, verbose = 1, validation_split = validation_split)

			with open("gru-2.txt", "a") as f:
				f.write(str(mem_size) + " " + str(embedding_dim) + " " + str(hist.history['val_loss'][0]))
				f.write("\n")
