import os

import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Dense

folder_path = 'fonts'

all_dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        all_dataframes.append(df)

df = pd.concat(all_dataframes, ignore_index=True)

label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
df['font'] = label_encoder1.fit_transform(df['font'])
df['fontVariant'] = label_encoder2.fit_transform(df['fontVariant'])

train = df.drop('font', axis=1)

train_data, test_data, train_labels, test_labels = train_test_split(
    train, df['font'], test_size=0.2, random_state=42
)

train_labels = to_categorical(train_labels, 153)
test_labels = to_categorical(test_labels, 153)

backend.clear_session()
model = Sequential()
model.add(Dense(128, input_dim=train_data.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(768, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(len(label_encoder1.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=20, batch_size=150, validation_split=0.2, verbose=1)
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')