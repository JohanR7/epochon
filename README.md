# Problem Statement

Agriculture faces significant challenges in optimizing crop yield and soil health, particularly with the growing need for sustainable and efficient farming practices. Farmers often lack access to precise, data-driven recommendations for selecting crops and fertilizers that align with environmental conditions, soil quality, and sustainability goals. Current solutions are either too generic or require expensive tools and expertise, making them inaccessible for small- and medium-scale farmers.

This gap results in suboptimal agricultural productivity, overuse of chemical fertilizers, degradation of soil quality, and increased environmental impact. To address these challenges, Earthworm provides an easy-to-use, intelligent system that delivers tailored recommendations for crop selection and fertilizer application based on key input constraints such as soil properties, weather conditions, and sustainability metrics.
# Problem Overview

![image](https://github.com/user-attachments/assets/c153b894-de12-4069-b9c0-7a7b9eceea22)
<br/>


Earthworm is an innovative platform designed to empower farmers by providing tailored crop and fertilizer recommendations based on their soil’s nutrient composition and environmental conditions. By integrating advanced data analytics and AI-driven algorithms, Earthworm analyzes soil values such as pH levels, macronutrients, and micronutrients, along with regional factors like rainfall and climate. The platform generates actionable insights to help farmers maximize yield, reduce costs, and ensure sustainable agricultural practices. Whether you’re managing a small farm or a large agricultural enterprise, Earthworm serves as a reliable partner in optimizing crop selection and fertilizer application for healthier soil and better productivity.
<br/>



Earthworm AI, the platform’s interactive chatbot, serves as a virtual assistant to guide farmers in making informed decisions. Powered by Google's Gemini, it provides instant, user-friendly support by answering questions, analyzing soil values, and offering tailored advice on crop and fertilizer selection. Farmers can simply input their queries or soil data, and Earthworm AI delivers actionable insights, making the process seamless and accessible. Beyond recommendations, the chatbot helps educate users about sustainable practices and optimizes resource use, enhancing productivity and environmental care.
<br/>

# Project Working


## Data

The data used to train the model was collected from the [Crop Prediction](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) dataset. The dataset consists of 2200 samples of 22 different crops whose predictions are made using 7 features: nitrogen, phosphorus, potassium, and pH content of the soil, temperature, humidity and rainfall. The dataset is perfectly balanced, with each crop having 100 samples.  The data was pre-processed to ensure consistency and cleaned to remove any missing values. The data includes information on various crop types, weather patterns, and soil types. 
<br/>
The data used to train the model was collected from the [Fertiliser Prediction](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction) dataset. The dataset consists of 100 entries of samples of 7 different fertilisers whose predictions are made using 6 features nitrogen, phosphorus, potassium, temperature, humidiyt and moisture%




## Model

The  models are built using Deep Neural Networks(DNNs). 
<br/>
The architecture we have chosen for crop prediction consists of 3 hidden layers with 128, 64 and 128 neurons respectively, and an output layer of 22 neurons, each corresponding to one type of crop. The activation function in the input and hidden layers is SeLU, while the activation function for the output layer is softmax. 
<br/>
![Screenshot 2024-12-01 at 3 59 51 PM](https://github.com/user-attachments/assets/7c58c124-a1c5-4baf-a5c3-b1081ba67dd7)
<br/>



The architecture chosen for fertiliser prediction consistes of two layers with 128 and 64 neurons respectively and an output layer of 7 neurons each corresponding to each fertiliser. The activation function used in hidden layers is SeLU and softmax in the output layer.
<br/>
![Screenshot 2024-12-01 at 4 00 22 PM](https://github.com/user-attachments/assets/fff72f25-02e7-425b-a027-ad7e2f12c39a)
<br/>
The model was created using Tensorflow and Keras.

## Training

The model was trained on the data with a 80:20 train-test split ratio. The objective function used was categorical crossentropy and the optimizer used was adam. Accuracy was used as the metric to evaluate the performance of the models. Training was done for 100 epochs on the crop prediction model and 50 epochs on fertiliser prediction model .


## Results

For crop prediction, our model got a 99% accuracy on train data and 97% accuracy on test data.
<br/>
![Screenshot 2024-12-01 at 6 43 41 PM](https://github.com/user-attachments/assets/1591ed7c-a054-4e5e-b5cb-4a812942dbba)
<br/>

For fertiliser prediction our model got a 100% accuracy on both train and accuracy data.
<br/>
![Screenshot 2024-12-01 at 6 44 23 PM](https://github.com/user-attachments/assets/a2ca48d7-8f71-4d4d-ae2f-152a1b12de48)
<br/>

## Chatbot

We have implemented a chatbot using Google's Gemini model


## Working

The following data is collected from the end-user to make predictions:
<br/>
N, P, K, pH, temperature, humidity, moisture%, rainfall, location.
<br/>
Further, we have created an interactive interface for users to enter relevant data to get predictions for their crop.
<br/>
![image](https://github.com/user-attachments/assets/ea5d2f40-022c-4b27-9f8a-148903366843)
<br/>
![image](https://github.com/user-attachments/assets/e561e8e7-67bb-4548-8dde-1c6eea01fed7)
<br/>
![image](https://github.com/user-attachments/assets/1859300a-e781-4ae6-95af-604f54225a7c)
<br/>
![image](https://github.com/user-attachments/assets/1df9f58a-d851-49be-aaef-f9add837f371)
<br/>
![image](https://github.com/user-attachments/assets/fcbda7b4-49e5-4ccf-be51-cffa27cbcb2a)
<br/>

# Tech Stack
Python using Tensorflow and Keras
<br/>
flask
<br/>
HTML CSS
<br/>

# Installation 
Clone the repository
<br/>
Create virtual environment: python -m venv venv
<br/>
Activate virtual environment
<br/>
venv/Script/activate
<br/>
Install dependencies:
<br/>
pip install requirements.txt
<br/>



## Team Members

PowerRangers: Codeforce
<br />
Aman Backer
<br/>
Johan Rajesh
<br/>
Advaith Dinkar
<br/>

A Vyshak
