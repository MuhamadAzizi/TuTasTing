import os
import cv2
import numpy as np
from tensorflow.keras import models
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DEBUG'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = models.load_model('my_model')

def encoder(y_pred):
    if y_pred == 0:
        return 'Kertas'
    elif y_pred == 1:
        return 'Batu'
    elif y_pred == 2:
        return 'Gunting'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

        file = request.files['file']
        filename = secure_filename(file.filename)
        path_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(path_file))

        img = cv2.imread(path_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (125, 125))

        x = np.array(img).reshape(1, 125, 125, 1).astype('float32') / 255.0
        prob = model.predict(x)
        y_pred = np.argmax(prob)
        prob_batu, prob_kertas, prob_gunting = prob[0][1], prob[0][0], prob[0][2]

        return render_template('predict.html', y_pred=encoder(y_pred), img=f'uploads/{filename}', \
            batu='{:.2f}'.format(prob_batu*100),
            kertas='{:.2f}'.format(prob_kertas*100),
            gunting='{:.2f}'.format(prob_gunting*100)
        )

if __name__ == '__main__':
    app.run()