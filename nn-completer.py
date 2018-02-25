#!/usr/bin/env python3


## === IMPORTS ===
# Suppress Keras messages by temporariy redirecting 
# stderr into the void
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import pickle
import numpy as np
from keras.models import load_model

sys.stderr = stderr


## === FUNCTIONS ===

def print_help():
    print("Usage: ./nn-completer.py <token> [<second token>]")


## === MAIN ===

def main():
    if len(sys.argv) == 1:
        print_help()
        exit(0)

    with open("python-final-w2n.dict", "rb") as f:
        w2n_dict = pickle.load(f)

    with open("python-final-n2w.dict", "rb") as f:
        n2w_dict = pickle.load(f)

    if len(sys.argv) == 2:
        model = load_model("python-final-model-single.h5")
    else:
        model = load_model("python-final-model-dual.h5")

    tokens = [ [] ]

    for t in sys.argv[-2:]:
        token = t if sys.argv[1] in w2n_dict.keys() else "<ID>"
        tokens[0].append(w2n_dict[token])

    net_input = np.array(tokens)
    net_output = model.predict(net_input)

    guesses = net_output.argsort()[0][::-1]

    for g in guesses[:5]:
        print(n2w_dict[g])

if __name__ == "__main__":
    main()
