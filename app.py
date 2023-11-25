# app.py
from flask import Flask, render_template, request, jsonify
import librosa
import joblib
import numpy as np

app = Flask(__name__)

# Load SVM model
with open('svm_model.joblib', 'rb') as model_file:
    svm_model = joblib.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get file from the request
    file = request.files['file']

    # Convert the file using librosa
    # You can modify this part based on your model input requirements
    # This is just a basic example
    data, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    # mfccs_processed = np.mean(mfccs.T, axis=0)

    # Use the loaded SVM model for prediction
    result = svm_model.predict(mfccs)

    # Return the result as JSON
    return jsonify({'result': result[0]})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
