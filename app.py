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

MODEL_PATH = 'model/'
model_name = 'markov'
model_file = f'{model_name}.model.json'
weights_file = f'{model_name}.weights.h5'

def load_json_wih_weights_to_model():
    json_file = open(f'{MODEL_PATH}{model_name}/{model_file}', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
	
    global loaded_model
    
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f'{MODEL_PATH}{model_name}/{weights_file}')
	
    print('[INFO] Loaded \'%s\' model with weights from disk' % model_name)
    
    global graph
    graph = tf.get_default_graph()
	
    return loaded_model
	
with open(f'{MODEL_PATH}tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    print('[INFO] Loaded Tokenizer from disk')	

classes_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	
def prepare_comment(comment):
    maxlen = 180
    tokenized_comment = tokenizer.texts_to_sequences(comment)
	
    return pad_sequences(tokenized_comment, maxlen=maxlen)
	
app = Flask(__name__)	
	
@app.route("/")
def hello():
    return "Welcome to Detox API!"	

@app.route('/detox/api/comment', methods=['POST'])
def predict():
    try:
        comment = request.json['comment']
        print('[INFO] Received request \'%s\'' % comment)

        # Tokenize the user's query and make a prediction
        with graph.as_default():
            prediction = loaded_model.predict([prepare_comment(comment)], batch_size=1024)

        output = {}
        i = 0
        # Assign scores to classes
        for label in classes_names:
            output[label] = '{0:.2f}'.format(prediction[0][i])
            i += 1
            
        return jsonify(output)
        
    except:
    
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    loaded_model = load_json_wih_weights_to_model()
    app.run(debug=True, port=8585)