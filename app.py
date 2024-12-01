from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PKL_DIR = os.path.join(MODEL_DIR, "pkl_files")
NORMALIZATION_DIR = os.path.join(MODEL_DIR, "normalization")

# Load Models and Files
model_path = os.path.join(MODEL_DIR, "2024-12-01_09-54-32-crop.h5")  # Change to your file name
model = load_model(model_path)

fertilizer_model_path = os.path.join(MODEL_DIR, "2024-12-01_09-51-29-fertilizer.h5")  # Change to your file name
fertilizer_model = load_model(fertilizer_model_path)

# Load Label Encoders
with open(os.path.join(PKL_DIR, "encoder.pkl"), 'rb') as file:
    encoder = pickle.load(file)

with open(os.path.join(PKL_DIR, "fertilizer_encoder.pkl"), 'rb') as file:
    fertilizer_encoder = pickle.load(file)

# Load Normalization Parameters
crop_normalization = np.load(os.path.join(NORMALIZATION_DIR, "normalization.npz"))
crop_scaler = StandardScaler()
crop_scaler.mean_ = crop_normalization['mean']
crop_scaler.scale_ = crop_normalization['std']

fertilizer_normalization = np.load(os.path.join(NORMALIZATION_DIR, "normalization1.npz"))
fertilizer_scaler = StandardScaler()
fertilizer_scaler.mean_ = fertilizer_normalization['mean']
fertilizer_scaler.scale_ = fertilizer_normalization['std']

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse input data
        data = request.json
        print("Received data:", data)

        # Crop Prediction
        crop_input_features = np.array([[float(data.get(key)) for key in [
            'nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'
        ]]])
        crop_input_scaled = crop_scaler.transform(crop_input_features)
        crop_predictions = model.predict(crop_input_scaled)
        crop_class_index = np.argmax(crop_predictions, axis=1)
        crop = encoder.inverse_transform(crop_class_index)[0]

        # Fertilizer Prediction
        fertilizer_input_features = np.array([[float(data.get(key)) for key in [
            'nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'moisture'
        ]]])
        fertilizer_input_scaled = fertilizer_scaler.transform(fertilizer_input_features)
        fertilizer_predictions = fertilizer_model.predict(fertilizer_input_scaled)
        fertilizer_index = np.argmax(fertilizer_predictions, axis=1)[0]
        fertilizer = fertilizer_encoder.inverse_transform([fertilizer_index])[0]

        return jsonify({"crop": crop, "fertilizer": fertilizer})

    except KeyError as e:
        print(f"Missing key: {e}")
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except ValueError as e:
        print(f"Invalid value: {e}")
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True, use_reloader=False)
