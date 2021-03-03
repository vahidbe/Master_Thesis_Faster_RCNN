import tensorflow as tf
import keras
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
gpu_available = tf.test.is_gpu_available()
built_with_cuda = tf.test.is_built_with_cuda()
print("=== Tensorflow version: " + str(tf.__version__))
print("=== Keras version: " + str(keras.__version__))
print("=== Gpu available: " + str(gpu_available))
print("=== Built with cuda: " + str(built_with_cuda))
