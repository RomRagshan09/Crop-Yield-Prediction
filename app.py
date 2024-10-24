from flask import Flask, request, render_template
import numpy as np
import pickle

# Load models
knr_model = pickle.load(open('Knr.pkl', 'rb'))  # KNeighborsRegressor model
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))  # Preprocessing pipeline

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        Crop = request.form['Crop']
        Crop_Year = request.form['Crop_Year']
        Season = request.form['Season']
        State = request.form['State']
        Area = request.form['Area']
        Annual_Rainfall = request.form['Annual_Rainfall']
        Fertilizer = request.form['Fertilizer']
        Pesticide = request.form['Pesticide']

        # Create an array of inputs
        features = np.array([[Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide]])
        
        # Preprocess the features
        transformed_features = preprocessor.transform(features)
        
        # Make the prediction
        predicted_value = knr_model.predict(transformed_features).reshape(1, -1)

        # Render the result in the template
        return render_template('index.html', prediction=predicted_value[0][0])

if __name__ == "__main__":
    app.run(debug=True)
