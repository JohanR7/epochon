import os
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Base directory (use script location or allow override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PKL_DIR = os.path.join(MODEL_DIR, "pkl_files")
NORMALIZATION_DIR = os.path.join(MODEL_DIR, "normalization1")

# Ensure directories exist
os.makedirs(PKL_DIR, exist_ok=True)
os.makedirs(NORMALIZATION_DIR, exist_ok=True)

# Load the dataset
csv_path = os.path.join(DATA_DIR, "Fertilizer Prediction 1.csv")
df = pd.read_csv(csv_path)

# Separate features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
num_classes = len(np.unique(labels))

# Save the encoder
encoder_path = os.path.join(PKL_DIR, "fertilizer_encoder.pkl")
with open(encoder_path, "wb") as file:
    pickle.dump(encoder, file)

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Save normalization parameters
normalization_path = os.path.join(NORMALIZATION_DIR, "normalization1.npz")
np.savez(normalization_path, mean=scaler.mean_, std=np.sqrt(scaler.var_))

# Split the data
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the model
fertilizer_model = Sequential([
    Dense(128, activation='selu', input_shape=(features.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='selu'),
    Dense(num_classes, activation='softmax')  # Output layer for Fertilizer
])

# Compile the model
fertilizer_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
EPOCHS = 50
fertilizer_history = fertilizer_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=EPOCHS, batch_size=32, verbose=2)

# Save the model with a timestamped name
model_name = os.path.join(BASE_DIR, "model", "2024-12-01_09-51-29-fertilizer.h5")
model_path = os.path.join(MODEL_DIR, model_name)
fertilizer_model.save(model_path)

# Plot training and validation loss
plt.plot(fertilizer_history.history['loss'], label='Train Loss')
plt.plot(fertilizer_history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Fertilizer Model - Loss')
plt.show()

# Evaluate the model
fertilizer_train_loss, fertilizer_train_acc = fertilizer_model.evaluate(train_x, train_y, verbose=0)
fertilizer_val_loss, fertilizer_val_acc = fertilizer_model.evaluate(val_x, val_y, verbose=0)

print(f"Fertilizer Model Accuracy on Training Set: {fertilizer_train_acc * 100:.2f}%")
print(f"Fertilizer Model Accuracy on Validation Set: {fertilizer_val_acc * 100:.2f}%")
