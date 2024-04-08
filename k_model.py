import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential, save_model
from keras.layers import LSTM, Dropout, Dense

import matplotlib.pyplot as plt

# Step 1: Read CSV file and preprocess the data
df = pd.read_csv('b_data.csv').dropna(axis=0)
df['Date'] = pd.to_datetime(df['Date'], utc=True)

# Step 2: Prepare the data
sc = MinMaxScaler(feature_range=(0,1))
df_scaled = sc.fit_transform(df.iloc[:, 1:2])

#X_train, X_test, y_train, y_test = train_test_split(df_scaled, df_scaled, test_size=0.2, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df_scaled[:, 0], test_size=0.2, shuffle=True, random_state=42)

# Step 3: Generate input sequences
def generate_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 90  # Define the sequence length
X_train, y_train = generate_sequences(X_train, seq_length)
X_test, y_test = generate_sequences(X_test, seq_length)

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 4: Define the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    Dense(units=1)
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32)

# Step 7: Evaluate the model
train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)

print(f'Training Loss: {train_loss}')
print(f'Test Loss: {test_loss}')


save_model(model , 'final_model.keras')

# Preprocess the new data
new_data = pd.read_csv('b_data.csv').dropna(axis=0)
new_data['Date'] = pd.to_datetime(new_data['Date'], utc=True)
new_data_scaled = sc.transform(new_data.iloc[:, 1:2])

# Check if the length of new_data_scaled is greater than or equal to seq_length
if len(new_data_scaled) >= seq_length:
    # Generate input sequences
    X_new, _ = generate_sequences(new_data_scaled, seq_length)

    # Reshape the data for LSTM
    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

    # Use the model to make predictions
    predicted_prices = model.predict(X_new)


    # Inverse transform the predicted prices
    predicted_prices = sc.inverse_transform(predicted_prices.reshape(-1,1))

    # Print or display the predicted prices
    for i, price in enumerate(predicted_prices):
        print(f"Predicted Price for {new_data['Date'].iloc[i+seq_length-1]}: {price[0]}")
else:
    print("Error: Insufficient data to generate sequences.")