import h5py
from flask import Flask, request, render_template
from keras.models import load_model
from classify import c100_classify
from scipy import misc
from skimage import io
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def entry_page():
    return render_template('index.html')

@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    saved_model = '/Users/andrew/Desktop/cifar/saved_models/cifar100.h5'
    model = load_model(saved_model)
    
    try:
        image_url = request.form['image_url']
        image = io.imread(image_url)
        image_small = misc.imresize(image,(32,32,3))/255.
        pred = c100_classify(image_small, model)
            message = "OK, here's what I think this is:"
    except:
        message = "Something has gone completely wrong, what did you do?!  Try another image."
    data = [{'name':x, 'probability':y} for x,y in zip(pred.iloc[:,1],pred.iloc[:,0])]

    return render_template('index.html',
                            message=message,
                           data=data)


if __name__ == '__main__':
    app.run(debug=True)
