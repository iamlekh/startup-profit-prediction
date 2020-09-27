from flask import Flask , render_template, request
from flask import Flask, render_template, request, session, redirect
import numpy as np
import joblib

app = Flask(__name__) 

model = open("model.pkl", 'rb')
model_final = joblib.load(model)

@app.route('/') 
def test(): 
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST']) 
def prediction():
    if request.method == 'POST':
        try:
            NewYork = float(request.form['NewYork'])
            California = float(request.form['California'])
            Florida = float(request.form['Florida'])
            RnDSpend = float(request.form['RnD_Spend'])
            AdminSpend = float(request.form['Admin_Spend'])
            MarketSpend = float(request.form['Market_Spend'])
            pred_arg = [NewYork, California, Florida, RnDSpend, AdminSpend, MarketSpend]
            test_data_arr= np.array(pred_arg)
            test_data_num= test_data_arr.reshape(1,-1)
            prediction = model_final.predict(test_data_num)
            output = round(float(prediction),2)
        except ValueError:
            return "Please check input values."
        return render_template('pred.html', prediction = output)
if __name__ == '__main__': 
    app.run(host='0.0.0.0')
