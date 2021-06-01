from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/') # Homepage
def home():
    return render_template('index.html')


#@app.route('/', methods=['GET', 'POST'])
@app.route('/predict',methods=['POST'])
def predict():
    housing = pd.read_csv("../data/modified_data.csv") ### change directory path if necessary
    housing['city'] = housing['city'].map(lambda x: str(x)[:-4])
    housing['city2'] = housing['city']
    house = housing.groupby('city2').first()

    if request.method == 'POST':
        from sklearn.externals import joblib
        model = joblib.load('gbr.ml')
    
        user_city_code = request.form.get('city_code')
        user_bed = request.form.get('bed')
        user_bath = request.form.get('bath')
        user_sqrt = request.form.get('sqrt')
        
        if request.form.get('city_code') == '':
            return render_template('index.html', predicts='NOT AVAILABLE.', enters=f'City Name: {user_city_code}, Bed: {user_bed}, Bath: {user_bath}, Squared feet: {user_sqrt}')

        if request.form.get('user_bed') == '':
            return render_template('index.html', predicts='NOT AVAILABLE.', enters=f'City Name: {user_city_code}, Bed: {user_bed}, Bath: {user_bath}, Squared feet: {user_sqrt}')
        
        if request.form.get('user_bath') == '':
            return render_template('index.html', predicts='NOT AVAILABLE.', enters=f'City Name: {user_city_code}, Bed: {user_bed}, Bath: {user_bath}, Squared feet: {user_sqrt}')
        
        if request.form.get('user_sqrt') == '':
            return render_template('index.html', predicts='NOT AVAILABLE.', enters=f'City Name: {user_city_code}, Bed: {user_bed}, Bath: {user_bath}, Squared feet: {user_sqrt}')
                
        
        init_features = [x for x in request.form.values()]

        if user_city_code not in house['city'].values:
            return render_template('index.html', predicts='NOT AVAILABLE.', enters=f'City Name: {user_city_code}, Bed: {user_bed}, Bath: {user_bath}, Squared feet: {user_sqrt}')
        
        for i in range(house.shape[0]):
            if house['city'][i] == user_city_code:
                init_features[0] = house['city_code'][i]
                break
               
        init_features = [float(x) for x in init_features]
        final_features = [np.array(init_features)]
            
        prediction = model.predict(final_features)
        
    if prediction[0] == 0:
        return render_template('index.html', predicts='under $499,999 US dollars.', enters=f'City Name: {user_city_code}, Bed: {str(final_features[0][1])}, Bath: {str(final_features[0][2])}, Squared feet: {str(final_features[0][3])}')
    elif prediction[0] == 1:
        return render_template('index.html', predicts='between $500,000 and $999,999 US dollars.', enters=f'City Name: {user_city_code}, Bed: {str(final_features[0][1])}, Bath: {str(final_features[0][2])}, Squared feet: {str(final_features[0][3])}')
    else:
        return render_template('index.html', predicts='above $1,000,000 US dollars.', enters=f'City Name: {user_city_code}, Bed: {str(final_features[0][1])}, Bath: {str(final_features[0][2])}, Squared feet: {str(final_features[0][3])}')


if __name__ == '__main__':
    app.run(debug=True)