import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle
import datetime




df = pd.read_csv("C:\\Users\\johan\\Desktop\\test_hackathon\\crop_pred\\data\\Crop_recommendation.csv")

features=df.iloc[:,:-1]
labels=df.iloc[:,-1]

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
num_classes = len(np.unique(labels))

os.makedirs('C:\\Users\\johan\\Desktop\\test_hackathon\\crop_pred\\model\\pkl_files', exist_ok=True)
with open("C:\\Users\\johan\\Desktop\\test_hackathon\\crop_pred\\model\\pkl_files\\encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)


scaler = StandardScaler()
features = scaler.fit_transform(features)

os.makedirs('C:/Users/johan/Desktop/test_hackathon/crop_pred/model/normalization', exist_ok=True)
np.savez("C:/Users/johan/Desktop/test_hackathon/crop_pred/model/normalization/normalization.npz", mean=scaler.mean_, std=np.sqrt(scaler.var_))


train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=42) #shuffles and splits dataset


model=Sequential()
model.add(Dense(128, activation='selu', input_shape=(features.shape[1],)))
model.add(Dense(64, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])


EPOCHS = 100
history=model.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=EPOCHS, batch_size=32, verbose=2)

os.makedirs('C:/Users/johan/Desktop/test_hackathon/crop_pred/model', exist_ok=True)
model_name = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-') + '.h5'
model_path = f'./model/{model_name}'
model.save(model_path)


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

train_loss, train_acc = model.evaluate(train_x, train_y, verbose=0)
val_loss, val_acc=model.evaluate(val_x,val_y, verbose=0)

print(f"Accuracy on training set: {train_acc * 100:.2f}%")
print(f"Accuracy on validation set: {val_acc * 100:.2f}%")


                                                                                        





