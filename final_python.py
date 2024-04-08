# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential, save_model
from keras.layers import LSTM, Dropout, Dense

import matplotlib.pyplot as plt

# %%
df = pd.read_csv('p_data.csv').dropna(axis=0)
f_df = df[['Date','Price']]

# Assuming selected_df contains the 'Date' and 'Price' columns
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'][::-1], color='blue', linestyle='-')
plt.title('BTC Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
f_df = f_df['Price']

sc = MinMaxScaler(feature_range=(0,1))

f_df = sc.fit_transform(np.array(f_df).reshape(-1,1))
print(f_df.shape)

# %%
training_size=int(len(f_df)*0.60)
test_size=len(f_df)-training_size
train_data,test_data=f_df[0:training_size,:],f_df[training_size:len(f_df),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

# %%
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# %%
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# %%
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

# %%
model = Sequential([
    LSTM(10, input_shape=(None, 1), activation='relu'),
    Dense(1)
])

model.compile(loss="mean_squared_error",optimizer="adam")

# %%
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1)

# %%
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()

# %%
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict.shape, test_predict.shape

# %%
train_predict = sc.inverse_transform(train_predict)
test_predict = sc.inverse_transform(test_predict)
original_ytrain = sc.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = sc.inverse_transform(y_test.reshape(-1,1)) 

# %%
print(train_predict)

print('Now test data')
print(test_predict[:,0])



# %%

x_input=f_df[::-1][len(f_df)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

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

# %%
last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)
print(last_days)
print(day_pred)

# %%
temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = sc.inverse_transform(f_df[len(f_df)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = sc.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})


# %%


plt.figure(figsize=(10, 6))
plt.plot(new_pred_plot.index, new_pred_plot['last_original_days_value'], color='blue', label='Last Original Days Value')
plt.plot(new_pred_plot.index, new_pred_plot['next_predicted_days_value'], color='red', label='Next Predicted Days Value')
plt.title('BTC Price Prediction')
plt.xlabel('Time Step')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
from itertools import cycle

lstmdf=f_df[::-1].tolist()
lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
lstmdf=sc.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

names = cycle(['Close price'])

# %%

# Plot the data
plt.figure(figsize=(10, 6))
for name, data in zip(names, [lstmdf]):
    plt.plot(data, label=name)

plt.title('BTC Price Prediction')
plt.xlabel('Time Step')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Plot the data
plt.figure(figsize=(10, 6))

# Plot actual data
plt.plot(lstmdf[:len(f_df)], color='blue', label='Actual Price')

# Plot predicted data
plt.plot(range(len(f_df), len(lstmdf)), lstmdf[len(f_df):], color='red', label='Predicted Price')

plt.title('BTC Price Prediction')
plt.xlabel('Time Step')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# %% [markdown]
# here 1 timestep is equal to a 15 day interval of data, hence, 1 timestep = 15 days


