import os

import pandas as pd
from tensorflow.python.keras.layers import MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten

folder_path = 'fonts'

all_dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        all_dataframes.append(df)

df = pd.concat(all_dataframes, ignore_index=True)
df = df.drop(
    ['fontVariant', 'm_label', 'strength', 'italic', 'orientation', 'm_top', 'm_left', 'originalH', 'originalW',
     'h', 'w'], axis=1)

label_encoder1 = LabelEncoder()
df['font'] = label_encoder1.fit_transform(df['font'])

train = df.drop('font', axis=1)

train_data, test_data, train_labels, test_labels = train_test_split(
    train, df['font'], test_size=0.2, random_state=42
)

# Data normalization
train_data = train_data / 255.0
test_data = test_data / 255.0
train_data = train_data.values.reshape(-1, 20, 20, 1)
test_data = test_data.values.reshape(-1, 20, 20, 1)

train_labels = to_categorical(train_labels, 153)
test_labels = to_categorical(test_labels, 153)

backend.clear_session()

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=400, activation='relu'))
model.add(Dense(units=800, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=153, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.2)
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
model.save('fonts_classification_cnn_model.h5')
