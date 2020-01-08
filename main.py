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
    with open('base.csv', 'a+', encoding='utf-8') as base:
        base.writelines("batch_size,learning_rate,epoch,sample_size\n")
        base.writelines(str(batch_size)+","+str(lr)+","+str(epoch)+","+str(sample_size)+"\n")

def exec_resnet(augmentation=True):
    def func():
        resnet.bulk_train(epoches=25, augmentation=augmentation)
    exec_result = timeit.repeat(func, number=1, repeat=5)
    
    with open('resnet_'+augmentation+'_timing.log', 'a+', encoding='utf-8') as timing:
        timing.writelines(exec_result)
    
    with open("resnet18_"+str(augmentation)+".csv", 'a+', encoding="utf-8") as resnet18:
        resnet18.writelines("average_duration,average_accuracy,average_valid_accuracy,average_loss,average_valid_loss\n")
        resnet_log = pandas.read_csv('restnetTrue.log', names=['accuracy', 'val_accuracy', 'loss', 'val_loss'],)
        resnet18.writelines(str(mean(exec_result))+","+str(mean(resnet_log['accuracy'].values))+","+str(mean(resnet_log['val_accuracy'].values))+","+str(mean(resnet_log['loss'].values))+","+str(mean(resnet_log['val_loss'].values))+"\n")

def log_inceptionv3():
    def func():
        inception_v3.training(32, 0.0001, optimizers.Adam, 'categorical_crossentropy')
    exec_inv3 = timeit.repeat(func, number=1, repeat=5)
    
    with open("inceptionv3.csv", 'a+', encoding="utf-8") as log:
        log.write("average_duration,average_accuracy,average_valid_accuracy,average_loss,average_valid_loss\n")
        inceptionv3_log = pandas.read_csv('inception_v3.log',
                                      names=['accuracy', 'val_accuracy', 'loss', 'val_loss'])
        log.write(str(mean(exec_inv3))+","+str(mean(inceptionv3_log['accuracy'].values))+","+str(mean(inceptionv3_log['val_accuracy'].values))+","+str(mean(inceptionv3_log['loss'].values))+","+str(mean(inceptionv3_log['val_loss'].values))+"\n")


# base()
# log_inceptionv3()
exec_resnet()
# exec_resnet(augmentation=False)