IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING
1. Introduction
Iris Flower Classification is a popular machine learning project used to classify flowers into different species based on their features. The objective of this project is to predict the species of an iris flower using measurements like sepal length, sepal width, petal length, and petal width.

This is a multi-class classification problem where the model predicts one of three classes: Setosa, Versicolor, or Virginica.

2. Objective
To understand classification problems in machine learning
To implement a classification algorithm
To classify iris flowers based on their features
To build a predictive model
3. Technology Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
4. Dataset Description
The dataset contains:

Sepal Length
Sepal Width
Petal Length
Petal Width
Species (Target)
Classes:

Setosa
Versicolor
Virginica
5. Methodology
Step 1: Load Dataset
Step 2: Data Preprocessing
Step 3: Split Data (Train/Test)
Step 4: Train Model
Step 5: Predict Output
Step 6: Evaluate Accuracy
6. Algorithm Used
We use Logistic Regression / Classification Model for predicting flower species.

7. Implementation (Code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("iris_data.csv")

print("Dataset Preview:")
print(data.head())

# Features and target
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = data['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Custom Prediction
print("\nEnter Flower Details:")
sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

sample = [[sl, sw, pl, pw]]
result = model.predict(sample)

print("Predicted Species:", result[0])
8. Results
The model successfully classifies iris flowers into three categories. Accuracy is generally very high due to well-structured data.

9. Advantages
Simple and efficient
High accuracy
Easy to understand
10. Limitations
Works best with clean data
Limited to given features
11. Conclusion
This project demonstrates how machine learning can be used for classification problems. It helps in understanding how models can categorize data into multiple classes.

12. Future Scope
Use advanced models (SVM, Random Forest)
Deploy as web app
Use larger datasets
