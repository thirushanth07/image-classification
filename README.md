# image-classification
Deep Learning Image Classification using TensorFlow and Transfer Learning  This project implements an image classification system using Deep Learning techniques with TensorFlow and the CIFAR-10 dataset. The model uses a pretrained MobileNetV2 architecture with transfer learning to improve classification accuracy and reduce training time. 
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
