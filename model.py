# Iris Flower Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("iris_data.csv")

# Features and target
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = data['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Custom input
print("\nEnter flower details:")
sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

result = model.predict([[sl, sw, pl, pw]])

print("Predicted Flower:", result[0])