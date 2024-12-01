from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from threading import Thread

# Load Models and Files
print("Loading model...")
model_path = 'C:/Users/johan/Desktop/test_hackathon/crop_pred/model/2024-11-30-01-23-51-280919.h5'
model = load_model(model_path)
print("Model loaded successfully.")

print("Loading encoder...")
with open('C:/Users/johan/Desktop/test_hackathon/crop_pred/model/pkl_files/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
print("Encoder loaded successfully.")

print("Loading normalization parameters...")
normalization = np.load('C:/Users/johan/Desktop/test_hackathon/crop_pred/model/normalization/normalization.npz')
scaler = StandardScaler()
scaler.mean_ = normalization['mean']
scaler.scale_ = normalization['std']
print("Normalization parameters loaded successfully.")

# Flask App
app = Flask(__name__)

@app.route("/")
def index():
    print("Rendering index.html...")
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse Input
        print("Parsing input data...")
        data = request.json
        print(f"Received data: {data}")

        # Crop Prediction
        crop_input_features = np.array([[ 
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])
        print(f"Input features before scaling: {crop_input_features}")

        crop_input_scaled = scaler.transform(crop_input_features)
        print(f"Input features after scaling: {crop_input_scaled}")

        crop_predictions = model.predict(crop_input_scaled)
        print(f"Model predictions: {crop_predictions}")

        crop_class_index = np.argmax(crop_predictions, axis=1)
        print(f"Predicted class index: {crop_class_index}")

        crop = encoder.inverse_transform(crop_class_index)[0]
        print(f"Recommended Crop: {crop}")

        # Return JSON response
        return jsonify({"crop": crop})

    except KeyError as e:
        print(f"Missing key: {e}")
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except ValueError as e:
        print(f"Invalid value: {e}")
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)