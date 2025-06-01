from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sklearn
print(sklearn.__version__)
import pickle

# Loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))

# create flask app for instance
app = Flask(__name__)

# define a route and a view function
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def prediction():
    if request.method=='POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        transform_features = preprocesser.transform(features)
        prediction = dtr.predict(transform_features).reshape(-1,1)
        
        return render_template('index.html', prediction=prediction[0][0])
        



# Run the app if the file is executed
if __name__ == '__main__':
    app.run(debug=True)