from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

data = pd.read_csv("iris_data.csv")
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = data['Species']

model = LogisticRegression(max_iter=200)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('iris.html')

@app.route('/predict', methods=['POST'])
def predict():
    sl = float(request.form['sl'])
    sw = float(request.form['sw'])
    pl = float(request.form['pl'])
    pw = float(request.form['pw'])

    result = model.predict([[sl, sw, pl, pw]])
    return render_template('iris.html', prediction=result[0])

if __name__ == "__main__":
    app.run(debug=True)