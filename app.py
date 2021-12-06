from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)
dict = {0 : 'Normal', 1 : 'TBC'}
model = tf.keras.models.load_model('model.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(150,150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = preprocess_input(image)
    yhat = model.predict(image)
    if yhat[0][0]>0:
        desc = 'NORMAL'
    elif yhat[0][1]>0:
        desc = 'TUBERCULOSIS'
    # label = decode_predictions(yhat)
    # label = label[0][0]

    # classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    classification = '%s' % (desc)

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)