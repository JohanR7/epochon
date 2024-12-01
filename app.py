from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
generation_config = {
    "temperature": 0.7,  
    "top_p": 1,
    "top_k": 40,  
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

gemini = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

conversation_history = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PKL_DIR = os.path.join(MODEL_DIR, "pkl_files")
NORMALIZATION_DIR = os.path.join(MODEL_DIR, "normalization")

model_path = os.path.join(MODEL_DIR, "2024-12-01_09-54-32-crop.h5")  
model = load_model(model_path)

fertilizer_model_path = os.path.join(MODEL_DIR, "2024-12-01_09-51-29-fertilizer.h5")  
fertilizer_model = load_model(fertilizer_model_path)

with open(os.path.join(PKL_DIR, "encoder.pkl"), 'rb') as file:
    encoder = pickle.load(file)

with open(os.path.join(PKL_DIR, "fertilizer_encoder.pkl"), 'rb') as file:
    fertilizer_encoder = pickle.load(file)

crop_normalization = np.load(os.path.join(NORMALIZATION_DIR, "normalization.npz"))
crop_scaler = StandardScaler()
crop_scaler.mean_ = crop_normalization['mean']
crop_scaler.scale_ = crop_normalization['std']

fertilizer_normalization = np.load(os.path.join(NORMALIZATION_DIR, "normalization1.npz"))
fertilizer_scaler = StandardScaler()
fertilizer_scaler.mean_ = fertilizer_normalization['mean']
fertilizer_scaler.scale_ = fertilizer_normalization['std']

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/input-form")
def input_form():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        print("Received data:", data)

        crop_input_features = np.array([[float(data.get(key)) for key in [
            'nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'
        ]]])
        crop_input_scaled = crop_scaler.transform(crop_input_features)
        crop_predictions = model.predict(crop_input_scaled)
        crop_class_index = np.argmax(crop_predictions, axis=1)
        crop = encoder.inverse_transform(crop_class_index)[0]

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
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        user_message = request.json['message']
        
        ai_identity_context = (
            "You are Earth Worm AI, a virtual assistant created to help beginner farmers. "
            "Your purpose is to provide clear and helpful farming advice, including crop recommendations, "
            "fertilizer suggestions, and general farming tips. Speak as if you're guiding someone who is new to farming."
        )
        
        message_with_context = ai_identity_context + " User asked: " + user_message
        
        conversation_history.append({"role": "user", "parts": [user_message]})
        
        chat = gemini.start_chat(history=conversation_history)
        response = chat.send_message(message_with_context)
        
        conversation_history.append({"role": "model", "parts": [response.text]})
        
        return jsonify({
            'message': response.text,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'message': f"An error occurred: {str(e)}",
            'success': False
        })


@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    global conversation_history
    conversation_history = []
    return jsonify({'success': True, 'message': 'Chat reset successfully'})
if __name__ == "__main__":
    app.run(port=5003, debug=True, use_reloader=False)
