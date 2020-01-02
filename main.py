from os import path

from tensorflow.keras import optimizers, models

from libs import configure_gpu, post_processing
from network.tensor import inception_v3

configure_gpu
if path.exists("../inception_v3.hdf5"):
    history = models.load_model('../inception_v3.hdf5')
    post_processing.postprocess(history, "inception_v3")
else:
    inception_v3.training(8, 0.0001, optimizers.Adam, 'categorical_crossentropy')
