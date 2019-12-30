from tensorflow.keras import optimizers

from libs import configure_gpu
from network.tensor import inception_v3

configure_gpu
inception_v3.training(8, 0.0001, optimizers.Adam, 'categorical_crossentropy')
