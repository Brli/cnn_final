from os import path

from tensorflow.keras import optimizers, models

from libs import configure_gpu, post_processing
from network.tensor import inception_v4

configure_gpu
if path.exists("../inception_v4.hdf5"):
    history = models.load_model('../inception_v4.hdf5')
else:
    inception_v4.training(32, 0.0001, optimizers.Adam, 'categorical_crossentropy')
