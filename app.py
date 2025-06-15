# app.py
import os
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Configuration & Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'baccarat_lstm_rich_model.h5') 
SCALER_PATH = os.path.join(BASE_DIR, 'rich_model_scaler.pkl') # Path to the saved scaler
SEQUENCE_LENGTH = 10

label_map = {"Player": 0, "Banker": 1, "Tie": 2}
inv_label_map = {v: k for k, v in label_map.items()}

# --- Load Model and Scaler ---
model = None
scaler = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Rich LSTM model and data scaler loaded successfully.")
    # Warm up the model
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, 4)) # Samples, Timesteps, Features
    model.predict(dummy_input)
    print("✅ Model warmed up.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model or scaler. Error: {e}")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_rich_sequence', methods=['POST'])
def predict_rich_sequence():
    if not model or not scaler:
        return jsonify({'error': 'AI model or data scaler is not loaded.'}), 503

    try:
        # The frontend will now send a list of 10 hand objects
        sequence_data = request.json.get('sequence', [])

        if len(sequence_data) != SEQUENCE_LENGTH:
            return jsonify({'error': f'Exactly {SEQUENCE_LENGTH} hands are required.'}), 400

        # 1. Prepare the feature list from the input data
        feature_list = []
        for hand in sequence_data:
            winner_int = label_map[hand['winner']]
            features = [
                int(hand['player_total']),
                int(hand['banker_total']),
                int(bool(hand['natural_win'])),
                winner_int
            ]
            feature_list.append(features)

        # 2. Scale the features using the loaded scaler
        scaled_features = scaler.transform(feature_list)
        
        # 3. Reshape for the LSTM model
        X_pred = np.array([scaled_features]) # Wrap in a list to create a "batch" of 1

        # 4. Make prediction
        probabilities = model.predict(X_pred)[0]
        predicted_int = np.argmax(probabilities)
        prediction_label = inv_label_map[predicted_int]
        confidence = probabilities[predicted_int] * 100

        return jsonify({
            'prediction': prediction_label,
            'confidence': f"{confidence:.2f}%",
            'based_on': 'AI model (Rich LSTM)'
        })

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)