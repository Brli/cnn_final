from os import path
from tensorflow.keras import optimizers, models
from numpy import mean
from libs import configure_gpu, post_processing
from network.tensor import inception_v3, inception_v4
from network.torch import resnet
import timeit, pandas

# configure_gpu
batch_size=32
lr=1e-4
epoch=25
sample_size=100
# inception_v4.training(32, 0.0001, optimizers.Adam, 'categorical_crossentropy')
# inception_v3.training(32, 0.0001, optimizers.Adam, 'categorical_crossentropy')
def base():
    base = open("base.csv", 'a+')
    base.write("batch_size,learning_rate,epoch,sample_size\n")
    base.write(str(batch_size)+","+str(lr)+","+str(epoch)+","+str(sample_size)+"\n")
    base.close()

def exec_resnet(augmentation=True):
    def func():
        resnet.bulk_train(augmentation=augmentation)
    exec_resnet = timeit.repeat(func, repeat=5)
    resnet18 = open("resnet18_"+str(augmentation)+".csv", 'a+')
    resnet18.writelines("average_duration,average_accuracy,average_valid_accuracy,average_loss,average_valid_loss")
    resnet_log = pandas.read_csv('resnet.log',
                                names=['accuracy', 'val_accuracy', 'loss', 'val_loss'])
    resnet18.write(str(mean(exec_resnet))+","+str(mean(resnet_log['accuracy'].values))+","+str(mean(resnet_log['val_accuracy'].values))+","+str(mean(resnet_log['loss'].values))+","+str(mean(resnet_log['val_loss'].values))+"\n")
    resnet18.close()

def incv3():
    inception_v3.training(32, 0.0001, optimizers.Adam, 'categorical_crossentropy')

def log_inceptionv3():
    inceptionv3 = open("inceptionv3", 'a+')
    inceptionv3.write("average_duration,average_accuracy,average_valid_accuracy,average_loss,average_valid_loss\n")
    exec_inv3 = timeit.repeat(incv3, repeat=5)
    inceptionv3_log = pandas.read_csv('inception_v3.log',
                                names=['accuracy', 'val_accuracy', 'loss', 'val_loss'])
    inceptionv3.write(str(mean(exec_inv3))+","+str(mean(inceptionv3_log['accuracy'].values))+","+str(mean(inceptionv3_log['val_accuracy'].values))+","+str(mean(inceptionv3_log['loss'].values))+","+str(mean(inceptionv3_log['val_loss'].values))+"\n")
    inceptionv3.close()

exec_resnet()
exec_resnet(augmentation=False)