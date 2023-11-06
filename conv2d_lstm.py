import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/data/Copy of filtered_all_cyclone[4th august 2023].csv")

df2 = df.dropna(axis=1, how='all')

df2

year = pd.to_datetime(df2['Date'], format='%Y-%m-%d').dt.year

month = pd.to_datetime(df2['Date'], format='%Y-%m-%d').dt.month

day = pd.to_datetime(df2['Date'], format='%Y-%m-%d').dt.day

ee = pd.to_datetime(df2['Date'], format='%Y-%m-%d').dt.time

hour = pd.to_datetime(ee, format='%H:%M:%S').dt.hour

df2['Date'] = hour

df2

#select features and target
X = df2.iloc[:, :-2].values #for 6 hours prediction
y1 = (df2['lon+6'])
y2 = (df2['lat+6'])

Y = np.column_stack((y1, y2))

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)
print("-----------------------------------------------------------------")
print(Y)

# Splitting without randomize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
k = 2852 #index for 55'th cyclone

X_train = X[:k,:]
X_test = X[k:,:]

Y_train = Y[:k]
Y_test= Y[k:]

# Assuming you have a Pandas DataFrame named 'df'
# Shape of df: (num_samples, num_features)

# Convert DataFrame to NumPy array
data_array1 = X_train

# Reshape the 2D array to a 3D array for Conv1D
X_train1 = np.expand_dims(data_array1, axis=2)
# Shape of data_conv1d: (num_samples, num_features, 1)
# Assuming you have X_train1 as a 3D NumPy array with shape (num_samples, num_features, 1)
# Convert it to 4D for Conv2D
X_train2 = np.expand_dims(X_train1, axis=3)

# Assuming you have a Pandas DataFrame named 'df'
# Shape of df: (num_samples, num_features)

# Convert DataFrame to NumPy array
data_array22 = X_test

# Reshape the 2D array to a 3D array for Conv1D
X_test1 = np.expand_dims(data_array22, axis=2)
# Shape of data_conv1d: (num_samples, num_features, 1)
X_test2 = np.expand_dims(X_test1, axis=3)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, LSTM, Dense, Reshape
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization

import tensorflow as tf
from tensorflow.keras import layers, models

def build_conv2d_lstm_model(input_shape):
    model = models.Sequential(name="model_conv2d_lstm")

    # Add Conv2D layers
    model.add(layers.Conv2D(12, (3, 1), activation='relu', input_shape=input_shape, name="Conv2D_1"))
    model.add(layers.MaxPooling2D((1, 1), name="MaxPooling2D_1"))
    model.add(layers.Conv2D(6, (3, 1), activation='relu', name="Conv2D_2"))
    model.add(layers.MaxPooling2D((1, 1), name="MaxPooling2D_2"))

    # Reshape for LSTM
    model.add(layers.Reshape((-1, 6)))  # Reshape to (timesteps, features)

    # Add Bidirectional LSTM layers
    model.add(layers.Bidirectional(layers.LSTM(6, return_sequences=True), name="Bidirectional_LSTM_1"))
    model.add(layers.Bidirectional(layers.LSTM(6, return_sequences=False), name="Bidirectional_LSTM_2"))

    model.add(layers.Flatten(name="Flatten"))

    # Add Dense layers
    model.add(layers.Dense(6, activation='linear', name="Dense_3"))
    model.add(layers.Dense(6, activation='linear', name="Dense_4"))
    model.add(layers.Dense(2, name="Dense_2"))

    return model

# Define input shape based on your data
input_shape = X_train2.shape[1:]

# Build the Conv2D-LSTM model
conv2d_lstm_model = build_conv2d_lstm_model(input_shape)
optimizer = tf.keras.optimizers.RMSprop(lr=1e-5, momentum=0.9)
# Compile the model and print the summary
conv2d_lstm_model.compile(optimizer=optimizer,
                          loss='mse',  # Adjust the loss function for your problem
                          metrics=['mae'])
conv2d_lstm_model.summary()

import keras.callbacks
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
# Store training stats
history = conv2d_lstm_model.fit(X_train2, Y_train, epochs=15, batch_size=32, verbose=1, validation_split=0.2)

Y_pred = conv2d_lstm_model.predict(X_test2)
Y_pred = Y_pred
print(Y_pred[:10])

lon_test = Y_test[:,0]
lat_test = Y_test[:,1]

lon_pred = Y_pred[:,0]
lat_pred = Y_pred[:,1]

random_dataset = pd.DataFrame()
random_dataset['lon_test'] = lon_test
random_dataset['lat_test'] = lat_test

random_dataset['lon_pred'] = lon_pred
random_dataset['lat_pred'] = lat_pred
random_dataset['deviation'] = np.sqrt(((lon_test-lon_pred)**2)+ ((lat_test-lat_pred)**2))

print("  ")
print("The average error for the deviation is "+ str(random_dataset['deviation'].mean()))
print("  ")

random_dataset

from matplotlib import pyplot as plt
plt.figure(figsize=(14,4))
plt.plot(lon_test, label='Lon Test')
plt.plot(lon_pred, label='Lon Predition')
plt.xlabel('Value index')
plt.ylabel('Longitude Value')
plt.ylim(80,100)
plt.title("Longitude prediction with ANN")
plt.legend()

plt.figure(figsize=(14,4))
plt.plot(lat_test, label='Lat Test')
plt.plot(lat_pred, label='Lat Predition')
plt.xlabel('Value index')
plt.ylabel('Latitude Value')
plt.ylim(5, 25)
plt.title("Latitude prediction with ANN")
plt.legend()

X_original_scale = sc.inverse_transform(X_test)
cyclone_number=X_original_scale[:,0]

lon_test = Y_test[:,0]
lat_test = Y_test[:,1]

lon_pred = Y_pred[:,0]
lat_pred = Y_pred[:,1]

random_dataset = pd.DataFrame()
random_dataset['cyclone_number'] = cyclone_number
random_dataset['lon_test'] = lon_test
random_dataset['lat_test'] = lat_test

random_dataset['lon_pred'] = lon_pred
random_dataset['lat_pred'] = lat_pred
random_dataset['deviation'] = np.sqrt(((lon_test-lon_pred)**2)+ ((lat_test-lat_pred)**2))

print("  ")
print("The average error for the deviation is "+ str(random_dataset['deviation'].mean()))
print("  ")

random_dataset

!apt-get install -qq libgdal-dev libproj-dev
#!pip install --no-binary shapely shapely
!pip install --no-binary shapely shapely --force
!pip install cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
fig, axs = plt.subplots(3, 3, figsize=(16, 18), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})

for i in range(3):
    for j in range(3):
        # Taking the cyclone number from input
        cyclone_number = i*3 + j + 40

        # Separating the data for the desired cyclone
        cyclone_data = random_dataset[random_dataset['cyclone_number'] == float(cyclone_number)]
        #cyclone_data = cyclone_data.drop(cyclone_data.index[0]) #removing the last row as it creates problem


        # Plotting the cyclone data
        axs[i,j].plot(cyclone_data['lon_test'], cyclone_data['lat_test'], '-o', markersize=2.5, transform=ccrs.PlateCarree())
        axs[i,j].plot(cyclone_data['lon_pred'], cyclone_data['lat_pred'], '-o', markersize=2.5, transform=ccrs.PlateCarree())
        axs[i,j].set_extent([70, 100, 5, 30], crs=ccrs.PlateCarree())
        axs[i,j].coastlines(linewidth=1.4)
        axs[i,j].gridlines(draw_labels=True)
        axs[i,j].set_title(f"Cyclone {cyclone_number}")
        #axs[i,j].legend(['Test', 'Prediction'], loc='upper left')

# Set the overall title for the subplot
fig.suptitle("6h Prediction no prev feature", fontsize=20, y=1)
fig.legend(['Test', 'Prediction'], loc='upper left')

# Adjust spacing between subplots
fig.tight_layout()

conv2d_lstm_model.save('myman1.h5')

"""# Run this cell for same initial point"""

import keras
from keras.models import load_model

model = load_model('/content/my_model1.h5')

import numpy as np

# Assuming you have new data 'X_new' in a NumPy array
Y_pred = model.predict(X_test)

# 'predictions' will contain the model's predictions for the new data

lon_test = Y_test[:,0]
lat_test = Y_test[:,1]

lon_pred = Y_pred[:,0]
lat_pred = Y_pred[:,1]

random_dataset = pd.DataFrame()
random_dataset['lon_test'] = lon_test
random_dataset['lat_test'] = lat_test

random_dataset['lon_pred'] = lon_pred
random_dataset['lat_pred'] = lat_pred
random_dataset['deviation'] = np.sqrt(((lon_test-lon_pred)**2)+ ((lat_test-lat_pred)**2))

print("  ")
print("The average error for the deviation is "+ str(random_dataset['deviation'].mean()))
print("  ")

random_dataset