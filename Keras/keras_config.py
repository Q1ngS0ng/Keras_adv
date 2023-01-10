from keras import losses, backend
import numpy as np
import cv2
import keras
import tensorflow as tf
from keras.datasets import mnist
from tqdm import tqdm, trange
lenet_minist_path = "../models/lenet_mnist.h5"