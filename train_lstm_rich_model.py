# train_lstm_rich_model.py (with corrected imports)
import json
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf # The only tensorflow import needed

# --- Configuration ---
SEQUENCE_LENGTH = 10
MODEL_FILENAME = 'baccarat_lstm_rich_model.h5'

# --- 1. Load Data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, 'converted_data.json')
try:
    with open(DATA_FILE_PATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
    exit()

# --- 2. Feature Engineering & Scaling ---
label_map = {"Player": 0, "Banker": 1, "Tie": 2}
num_classes = len(label_map)

feature_list = []
for row in data:
    features = [
        row['player_total'],
        row['banker_total'],
        int(row['natural_win']),
        label_map[row['winner']]
    ]
    feature_list.append(features)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(feature_list)

# --- 3. Create Sequences of Rich Features ---
sequences, labels = [], []
winner_list_int = [label_map[row['winner']] for row in data]

for i in range(len(scaled_features) - SEQUENCE_LENGTH):
    seq = scaled_features[i : i + SEQUENCE_LENGTH]
    sequences.append(seq)
    label = winner_list_int[i + SEQUENCE_LENGTH]
    labels.append(label)

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels, num_classes=num_classes) # Changed here

print(f"✅ Generated {len(X)} sequences.")
print(f"Shape of X (samples, timesteps, features): {X.shape}")
print(f"Shape of y (samples, classes): {y.shape}")

# --- 4. Build the LSTM Model ---
input_shape = (X.shape[1], X.shape[2])

print("\nBuilding Rich Feature LSTM model...")
model = tf.keras.models.Sequential([ # Changed here
    tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True), # Changed here
    tf.keras.layers.Dropout(0.3), # Changed here
    tf.keras.layers.BatchNormalization(), # Changed here
    
    tf.keras.layers.LSTM(32, return_sequences=False), # Changed here
    tf.keras.layers.Dropout(0.3), # Changed here
    tf.keras.layers.BatchNormalization(), # Changed here
    
    tf.keras.layers.Dense(16, activation='relu'), # Changed here
    
    tf.keras.layers.Dense(num_classes, activation='softmax') # Changed here
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. Train and Evaluate ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Rich LSTM model...")
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    validation_split=0.1,
                    verbose=1)

print("\nEvaluating model...")
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print(f"\n✅ Rich LSTM Model Accuracy: {accuracy_score(y_test_labels, y_pred) * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test_labels, y_pred, target_names=label_map.keys(), zero_division=0))

# --- 6. Save Model AND Scaler ---
model.save(MODEL_FILENAME)
joblib.dump(scaler, 'rich_model_scaler.pkl')
print(f"\n✅ Rich LSTM model saved as {MODEL_FILENAME}")
print(f"✅ Data scaler saved as rich_model_scaler.pkl")