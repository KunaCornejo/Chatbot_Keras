"""
1MENTOR Inc., 2023.

Developed by: ML Department.
"""

import json
import numpy as np
from tensorflow import keras
import colorama
from colorama import Fore, Style
import pickle

colorama.init()


with open("data/raw/intents.json") as file:
    data = json.load(file)


def chat():
    """Give responses according to inputs, only for testing purposes."""
    # load trained model
    model = keras.models.load_model('models/chat_model.h5')

    # load tokenizer object
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('models/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model\
            .predict(keras
                     .preprocessing
                     .sequence
                     .pad_sequences(tokenizer.texts_to_sequences([inp]),
                                    truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                if i['tag'] == "self-appointment":
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, "Andy")
                elif i['tag'] == "navigate":
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,
                          '<a href="https://www.1mentor.io/higher-ed"> 1. \
                            Higher Education</a>')
                else:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,
                          np.random.choice(i['responses']))


print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" +
      Style.RESET_ALL)
chat()
