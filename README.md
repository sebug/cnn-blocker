# Augmenting web browsing experience using machine learning
General concept: HTML to graphs, detect those subgraphs.

The workbook: https://colab.research.google.com/drive/1T_0nmG-Ll0zLVLst4HEVvJ7CUA8KLH8G

So their dataset is a graph of HTML nodes. First step was to calculate graph kernels with [GraKeL](https://github.com/ysig/GraKeL).

We try some basic techniques on the kernel data (SVM, random forest classifier).

## Graph Convolutional Network
Convolutional networks - focus on the neighbors. Works well with graphs as well.

Their sample uses [Keras](https://keras.io)

## Learning on the web
Uses WebGL to benefit from the integrated GPU for re-learning.


