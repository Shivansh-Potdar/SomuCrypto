from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import LSTM, Dropout, Dense

import matplotlib.pyplot as plt

@app.route("/pred/<coin_name>")
def pred(coin_name):
    df = pd.read_csv(f'data/{coin_name}_data.csv').dropna(axis=0)
    f_df = df[['Date', 'Price']]['Price']
    model = load_model(f'models/{coin_name}_model.keras')

    time_step = 15

    sc = MinMaxScaler(feature_range=(0,1))

    f_df = sc.fit_transform(np.array(f_df).reshape(-1,1))

    x_input=f_df[::-1][len(f_df)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 30
    while(i<pred_days):
        
        if(len(temp_input)>time_step):
            
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
                
    print("Output of predicted next days: ", len(lst_output))

    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)

    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = sc.inverse_transform(f_df[len(f_df)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = sc.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    return next_predicted_days_value[time_step+1:]

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

