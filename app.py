from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)
dict = {0 : 'Normal', 1 : 'TBC'}
model = tf.keras.models.load_model('model_capstone.h5')
model.make_predict_function()

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')
    
@app.route('/submit', methods=['GET','POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(150,150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    

    yhat = model.predict(image)
    if yhat==0:
        desc = 'NORMAL'
    else:
        desc = 'TUBERKULOSIS'
    
    classification = '%s' % (desc)

    return render_template('index.html', prediction=classification, image=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)