import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import *
def read_png(filename):
    return tf.image.decode_png(tf.io.read_file(filename))