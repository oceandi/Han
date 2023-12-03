from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__, static_folder='.', static_url_path='')

# Modeli yükle
model = joblib.load('text_classifier.model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predictions = model.predict([data['text']])
    return jsonify({'prediction': predictions[0]})

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
