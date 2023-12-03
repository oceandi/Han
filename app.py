from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Modeli yükle
model = joblib.load('text_classifier.model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predictions = model.predict([data['text']])
    return jsonify({'prediction': predictions[0]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)