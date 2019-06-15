# Color cmd output
from colorama import init
init()
from termcolor import colored

# Ignore all warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import h5py

# Dependencies
from flask import Flask, request, jsonify
import numpy as np
import traceback
import pickle
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Change logging level of TensorFlow
tf.logging.set_verbosity(tf.logging.ERROR)

model_name = 'markov-glove'
model_file = f'{model_name}.model.json'
weights_file = f'{model_name}.weights.h5'

def load_json_wih_weights_to_model():
    json_file = open(f'{model_file}', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
	
    global loaded_model
    
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'{weights_file}')
	
    print(colored('[INFO] Loaded \'%s\' model with weights from disk' % model_name, 'yellow'))
    
    global graph
    graph = tf.get_default_graph()
	
    return loaded_model
	
with open(f'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    print(colored('[INFO] Loaded Tokenizer from disk', 'yellow'))	

classes_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	
def prepare_comment(comment):
    maxlen = 180
    tokenized_comment = tokenizer.texts_to_sequences(comment)
	
    return pad_sequences(tokenized_comment, maxlen=maxlen)
	
app = Flask(__name__, static_folder="static")
	
@app.route('/')
@app.route('/index')
def index():
    return '''
<html>
    <head>
        <title>Detox API</title>
        <link rel="shortcut icon" href="https://img.icons8.com/cotton/64/000000/comments.png">
        <link href='https://fonts.googleapis.com/css?family=Roboto' rel='stylesheet'>
    </head>
    <style>
    body {
        font-family: 'Roboto';font-size: 22px;
    }
    </style>
    <body>
        <h1>Welcome to Detox API!</h1>
    </body>
</html>'''

@app.route('/detox/api/comment', methods=['POST'])
def predict():
    try:
        comment = request.json['comment']
        print(colored('[INFO] Received comment \'%s\'' % comment, 'yellow'))
        
        # Tokenize the user's query and make a prediction
        with graph.as_default():
            prediction = loaded_model.predict([prepare_comment([comment])], batch_size=1024)

        output = {}
        i = 0
        is_toxic = False
        # Assign scores to classes
        for label in classes_names:
            result = prediction[0][i]
            if result > 0.5:
                is_toxic = True
            output[label] = '{0:.2f}'.format(prediction[0][i])
            i += 1
        
        print(colored('Results:', 'cyan'))
        print(colored(output, 'cyan'))
        if is_toxic:
            print(colored('This comment seems to be toxic!', 'magenta'))
        else:
            print(colored('This comment seems to be harmless!', 'green'))
        return jsonify(output)
        
    except:
    
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    loaded_model = load_json_wih_weights_to_model()
    app.run(debug=True, host='0.0.0.0')
