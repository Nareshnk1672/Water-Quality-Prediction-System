from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('water_potability.csv')

df.drop_duplicates(inplace=True)
df.dropna(how='all', inplace=True)
idx1 = df.query('Potability == 1')['ph'][df.ph.isna()].index
df.loc[idx1, 'ph'] = df.query('Potability == 1')['ph'][df.ph.notna()].mean()
idx0 = df.query('Potability == 0')['ph'][df.ph.isna()].index
df.loc[idx0,'ph'] = df.query('Potability==0')['ph'][df.ph.notna()].mean()
idx1 = df.query('Potability == 1')['Sulfate'][df.Sulfate.isna()].index
df.loc[idx1, 'Sulfate'] = df.query('Potability == 1')['Sulfate'][df.Sulfate.notna()].mean()
idx0 = df.query('Potability == 0')['Sulfate'][df.Sulfate.isna()].index
df.loc[idx0,'Sulfate'] = df.query('Potability==0')['Sulfate'][df.Sulfate.notna()].mean()
idx1 = df.query('Potability == 1')['Trihalomethanes'][df.Trihalomethanes.isna()].index
df.loc[idx1, 'Trihalomethanes'] = df.query('Potability == 1')['Trihalomethanes'][df.Trihalomethanes.notna()].mean()
idx0 = df.query('Potability == 0')['Trihalomethanes'][df.Trihalomethanes.isna()].index
df.loc[idx0,'Trihalomethanes'] = df.query('Potability==0')['Trihalomethanes'][df.Trihalomethanes.notna()].mean()
df.loc[~df.ph.between(6.5, 8.5), 'Potability'] = 0
X = df.drop(['Potability'], axis = 1).values
y = df['Potability'].values
sc = StandardScaler()
X = sc.fit_transform(X)
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, random_state=42)
rf.fit(X, y)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction page route
@app.route('/predict', methods=[ 'POST'])
def predict():
    # Check if the method used in the request is 'POST'
    if request.method == 'POST':
        # Get the form data
        data = request.form
        ph = float(data['ph'])
        Hardness = float(data['Hardness'])
        Solids = float(data['Solids'])
        Chloramines = float(data['Chloramines'])
        Sulfate = float(data['Sulfate'])
        Conductivity = float(data['Conductivity'])
        Organic_carbon = float(data['Organic_carbon'])
        Trihalomethanes = float(data['Trihalomethanes'])
        Turbidity = float(data['Turbidity'])

    # Create a dataframe with the form data
    input_data = pd.DataFrame({'ph': [ph], 'Hardness': [Hardness], 'Solids': [Solids],
                               'Chloramines': [Chloramines], 'Sulfate': [Sulfate], 'Conductivity': [Conductivity],
                               'Organic_carbon': [Organic_carbon], 'Trihalomethanes': [Trihalomethanes],
                               'Turbidity': [Turbidity]})

    # Scale the input data
    input_data = sc.transform(input_data)

    # Make a prediction
    prediction = rf.predict(input_data)

    # Return the prediction result
    if prediction == 1:
        return render_template('result.html', result=' is potable.')
    else:
        return render_template('result.html', result=' is not potable.')

if __name__ == '__main__':
    app.run(debug=True)