import tensorflow as tf

# Code from: https://www.tensorflow.org/guide/gpu

def __init__(self, mem_buff = True, mem_limit = 0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], mem_buff)
            if mem_limit != 0 :
                tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)]
                )
            logical_gpus=tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
