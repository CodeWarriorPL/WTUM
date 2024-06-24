import os
from flask import Flask, send_from_directory
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import base64
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Konfiguracja Flask
server = Flask(__name__)

# Konfiguracja Dash
app = Dash(__name__, server=server)

# Wczytaj wcześniej zapisany model
model_path = 'model.keras'  # Ścieżka do zapisanego modelu
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")

model = load_model(model_path)

# Słownik mapujący indeksy klas na nazwy kategorii
class_mapping = {0: 'Bags', 1: 'Belts', 2: 'Bottomwear', 3: 'Eyewear', 4: 'Flip Flops', 
                 5: 'Fragrance', 6: 'Innerwear', 7: 'Jewellery', 8: 'Sandal', 9: 'Shoes', 
                 10: 'Topwear', 11: 'Wallets', 12: 'Watches'}

# Layout aplikacji Dash
app.layout = html.Div([
    html.H1("GUI z modelu klasyfikacji obrazów"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Przeciągnij i upuść lub ',
            html.A('wybierz obrazek')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='prediction')
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    # Konwersja obrazu z base64 na numpy array
    decoded = base64.b64decode(content_string)
    img = image.load_img(BytesIO(decoded), target_size=(160, 120))  # Dopasowanie rozmiaru obrazu do modelu

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizacja obrazu

    # Przewidywanie klasy
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_mapping[predicted_class_index]

    return html.Div([
        html.H5(filename),
        html.H6(f'Przewidywana klasa: {predicted_class}')
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              Input('upload-image', 'filename'))
def update_output(contents, filename):
    if contents is None:
        raise PreventUpdate
    else:
        return parse_contents(contents, filename)

# Funkcja uruchamiania serwera

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("3000"), debug=True)
