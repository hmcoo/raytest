import mxnet as mx
import gym
from mxnet.gluon.model_zoo import vision

def resnet():
    resnet50 = vision.resnet50_v1()
    resnet101 = vision.resnet101_v1()
    resnet152 = vision.resnet152_v1()

    return (resnet50, resnet101, resnet152)
    #https://medium.com/apache-mxnet/implementing-resnet-with-mxnet-gluon-and-comet-ml-for-image-classification-9bb4ad93a53f
    #https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/deploy/inference/image_classification_jetson.html

def inception():
    inception = vision.inception_v3()
    #https://mxnet.apache.org/versions/1.8.0/api/python/docs/_modules/mxnet/gluon/model_zoo/vision/inception.html
    #https://wikidocs.net/62213
    #https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/inception.py
def vgg():

    vgg11 = vision.vgg11()
    vgg16 = vision.vgg16()

    return  (vgg11, vgg16)
    #https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/vgg.py

# def taxi():
#     NotImplemented
#
# def frozenlake():
#     NotImplemented
#
# def cartpole():
#     NotImplemented
#
# def mountaincar():
#     NotImplemented