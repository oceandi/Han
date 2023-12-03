from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Modeli yükle
model = joblib.load('text_classifier.model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predictions = model.predict([data['text']])
    return jsonify({'prediction': predictions[0]})

if __name__ == '__main__':
    app.run(debug=True)