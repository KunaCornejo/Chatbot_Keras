"""
1MENTOR Inc., 2023.

Developed by: ML Department.
"""

# Import the required packages and dependencies

from waitress import serve
from flask import Flask, render_template, request
import json
import numpy as np
from tensorflow import keras
import pickle
import os

# Previous Configuration

PATH = os.path.join(os.getcwd(), 'data/raw', 'intents.json')
with open(PATH) as file:
    data = json.load(file)

# Download model

PATH = os.path.join(os.getcwd(), 'models', 'chat_model.h5')
model = keras.models.load_model(PATH)

# Load tokenizer object

PATH = os.path.join(os.getcwd(), 'models', 'tokenizer.pickle')
with open(PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object

PATH = os.path.join(os.getcwd(), 'models', 'label_encoder.pickle')
with open(PATH, 'rb') as enc:
    lbl_encoder = pickle.load(enc)


def chat(inp, model, tokenizer, lbl_encoder, max_len=20) -> str:
    """
    Give a string response according to a string input.

    The conversational Ai-Chatbot uses this function to obtain responses.

    Parameters
    ----------
    inp : Input String to be processed.
    model : Trained “Sequential” model class of Keras by using Neural Networks
    tokenizer : Text data corpus vectorized by using the “Tokenizer” class and
                it allows us to limit our vocabulary size up to some defined
                number.
    lbl_encoder : Provided by scikit-learn to convert the target labels into a
                  model understandable form.
    max_len =  Length used by “pad_sequences” method to get all the trained
               text sequences of the same size.

    Returns
    -------
    response : String text generated by neural network model
    """
    result = model\
        .predict(keras
                 .preprocessing
                 .sequence
                 .pad_sequences(tokenizer.texts_to_sequences([inp]),
                                truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    link_1 = "https://www.1mentor.io"
    link_2 = "https://www.1mentor.io/higher-ed"
    link_3 = "https://www.instagram.com/1mentor.inc/?hl=en"
    link_4 = "https://twitter.com/1mentorinc?lang=en"
    link_5 = "https://www.linkedin.com/company/1mentor?original_referer=https\
             %3A%2F%2Fwww.1mentor.io%2F"

    for i in data['intents']:
        if i['tag'] == tag:
            if i['tag'] == "self-appointment":
                response = "Andy"

            elif i['tag'] == "navigate":
                response = "You can navigate on the following links:"\
                        + '<div></div>'\
                        + f'<a href={link_1}> 1. Home</a>'\
                        + '<div></div>'\
                        + f'<a href={link_2}> 2. Higher Education</a>'\
                        + '<div></div>'\
                        + f'<a href={link_3}> 3. Instagram</a>'\
                        + '<div></div>'\
                        + f'<a href={link_4}> 4. Twitter</a>'\
                        + '<div></div>'\
                        + f'<a href={link_5}> 5. LinkedIn</a>'

            else:
                response = np.random.choice(i['responses'])
    return response

# Create the Flask app


app = Flask(__name__)
# Define the home page


@app.route('/')
def home():
    """Flask API."""
    return render_template('index.html')
    # return "<p>Developed by 1Mentor Inc. 2023</p>"

# Define the function to generate the chatbot response


@app.route('/get-response')
def get_bot_response():
    """Conversational Ai-Chatbot uses this function to process.."""
    generated_response = chat(request.args.get('query'),
                              model,
                              tokenizer,
                              lbl_encoder,
                              max_len=20)

    return generated_response


# Run the app
if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
    app.run(debug=True)
