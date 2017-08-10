# Keras Convolutional Neural Network for CIFAR-100

This repo shows a convolution neural network (CNN) built for the CIFAR-100 dataset to create an object recognizer.  The CNN is meant to require less memory so it can easily be trained on a g2.2xlarge AWS GPU.  The trained model is then used to create a Flask/D3 app that can read in images from a URL and predict what the object is.  More information about this CNN and the app can be found on [my website](https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100).

Here is a demonstration of the app, showing the CNN recognizes both the bridge and castle:

<p align="center">
<img src="https://raw.githubusercontent.com/andrewkruger/cifar100_CNN/master/app/app.gif">
</p>

