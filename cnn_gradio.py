import os

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf


def get_fonts_class(img):
    img = img.resize((20, 20))
    img = np.array(img).reshape((-1, 20, 20, 1))
    prediction = model.predict(img).tolist()[0]
    return class_names[np.argmax(prediction[0])]


folder_path = 'fonts'

all_dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        all_dataframes.append(df)

df = pd.concat(all_dataframes, ignore_index=True)

class_names = df['font'].unique()

model = tf.keras.models.load_model('fonts_classification_cnn_model.h5')

im = gr.Image(type="pil", image_mode='L')
iface = gr.Interface(
    fn=get_fonts_class,
    inputs=im,
    outputs=gr.Label(),
    examples=['foto.png']
)
iface.launch(share=True)
