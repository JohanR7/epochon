import os
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Base directory (use script location or allow override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PKL_DIR = os.path.join(MODEL_DIR, "pkl_files")
NORMALIZATION_DIR = os.path.join(MODEL_DIR, "normalization")

# Ensure directories exist
os.makedirs(PKL_DIR, exist_ok=True)
os.makedirs(NORMALIZATION_DIR, exist_ok=True)

# Load the dataset
csv_path = os.path.join(DATA_DIR, "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

# Separate features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
num_classes = len(np.unique(labels))

# Save the encoder
encoder_path = os.path.join(PKL_DIR, "encoder.pkl")
with open(encoder_path, "wb") as file:
    pickle.dump(encoder, file)

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Save normalization parameters
normalization_path = os.path.join(NORMALIZATION_DIR, "normalization.npz")
np.savez(normalization_path, mean=scaler.mean_, std=np.sqrt(scaler.var_))

# Split the data
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='selu', input_shape=(features.shape[1],)),
    Dense(64, activation='selu'),
    Dense(128, activation='selu'),
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
EPOCHS = 100
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=EPOCHS, batch_size=32, verbose=2)

# Save the model with a timestamped name
os.makedirs(MODEL_DIR, exist_ok=True)
model_name = os.path.join(BASE_DIR, "model", "2024-12-01_09-54-32-crop.h5")
model_path = os.path.join(MODEL_DIR, model_name)
model.save(model_path)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Crop Recommendation Model - Loss')
plt.show()

# Evaluate the model
train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
val_loss, val_acc = model.evaluate(val_x, val_y, verbose=0)

print(f"Accuracy on training set: {train_acc * 100:.2f}%")
print(f"Accuracy on validation set: {val_acc * 100:.2f}%")
