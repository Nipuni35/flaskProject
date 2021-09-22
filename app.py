from flask import Flask
from flask import request
from keras.models import load_model
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello Nipuni!'

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
    parameters = request.get_json(force=True)
    im = np.array(parameters['image'])
    im = im.astype("float32") / 255
    im = np.expand_dims(im, -1)[None]
    out = id2class[np.argmax(model.predict(im))]
    return out

if __name__ == '__main__':
    app.run('0.0.0.0')
