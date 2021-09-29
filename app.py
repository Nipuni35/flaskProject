import flask
import numpy as np
from PIL import Image
from flask import Flask
from flask import request
from keras.models import load_model

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return flask.render_template('prediction_input.html')


id2class = {0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot", }

model = load_model("vgg16_cats_vs_dogs.h5")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['img']
    img = Image.open(file).convert('L')
    img = img.resize((28, 28))
    im = np.array(img)
    im = im.astype("float32") / 255
    im = np.expand_dims(im, -1)[None]
    out = id2class[np.argmax(model.predict(im))]
    return out


if __name__ == '__main__':
    app.run('0.0.0.0')
