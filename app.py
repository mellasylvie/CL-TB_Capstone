from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(150,150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    pred = model.predict(image)
    if pred==0:
        desc = 'NORMAL'
    elif pred==1:
        desc = 'TUBERCULOSIS'

    classification = '%s' % (desc)

    return render_template('index.html', prediction=classification, image=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)