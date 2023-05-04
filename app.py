# Import the required packages
from flask import Flask, render_template, request
import time
import json 
import numpy as np
from tensorflow import keras
import colorama 
from colorama import Fore, Style
import pickle

# Previous Configuration

colorama.init()

with open("intents.json") as file:
    data = json.load(file)

# Download model

model = keras.models.load_model('chat_model.h5')

# Load tokenizer object

with open('tokenizer.pickle', 'rb') as handle:
     tokenizer = pickle.load(handle)

# Load label encoder object

with open('label_encoder.pickle', 'rb') as enc:
     lbl_encoder = pickle.load(enc)

def chat(inp, model, tokenizer, lbl_encoder, max_len=20):
    # load trained model
    # model = keras.models.load_model('chat_model.h5')

    # load tokenizer object
    # with open('tokenizer.pickle', 'rb') as handle:
    #    tokenizer = pickle.load(handle)

    # load label encoder object
    # with open('label_encoder.pickle', 'rb') as enc:
    #    lbl_encoder = pickle.load(enc)

    # parameters
    # max_len = 20
    
    result = model\
                 .predict(keras\
                 .preprocessing\
                 .sequence\
                 .pad_sequences(tokenizer.texts_to_sequences([inp]), 
                                truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            if i['tag'] == "self-appointment":
                 response = "Andy"

            elif i['tag'] == "navigate":
                response = "You can navigate on the following links:"\
                        +'<div></div>'\
                        +'<a href="https://www.1mentor.io"> 1. Home</a>'\
                        +'<div></div>'\
                        +'<a href="https://www.1mentor.io/higher-ed"> 2. Higher Education</a>'\
                        +'<div></div>'\
                        +'<a href="https://www.instagram.com/1mentor.inc/?hl=en"> 3. Instagram</a>'\
                        +'<div></div>'\
                        +'<a href="https://twitter.com/1mentorinc?lang=en"> 4. Twitter</a>'\
                        +'<div></div>'\
                        +'<a href="https://www.linkedin.com/company/1mentor?original_referer=https%3A%2F%2Fwww.1mentor.io%2F"> 5. LinkedIn</a>'

            else:
                 response = np.random.choice(i['responses'])
    return response
            

# Create the Flask app

app = Flask(__name__)

# Define the home page

@app.route('/')
def home():
    return render_template('index.html')
    # return "<p>Developed by 1Mentor Inc. 2023</p>"

# Define the function to generate the chatbot response

@app.route('/get-response')
def get_bot_response():
    generated_response = chat(request.args.get('query'), 
                              model, 
                              tokenizer, 
                              lbl_encoder, 
                              max_len=20)
    
    return generated_response

# Run the app
if __name__ == '__main__':
    app.run(debug=True)