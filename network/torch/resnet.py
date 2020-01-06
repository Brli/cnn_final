import pandas as pd
import numpy as np
from tqdm import tqdm

# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import time

from os import listdir
from os.path import isfile, isdir, join

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, BatchNorm1d, ELU
from torch.optim import Adagrad, Adam, SGD

# torchvision for pre-trained models
from torchvision import models
# from torchvision.models import resnet18, vgg16_bn

# modules that data augmentation needs
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data



traindict = {0 : 'buildings', 1 : 'forest', 2 : 'glacier', 3 : 'mountain', 4 : 'sea', 5 : 'street'}
traindict



n_img = 75
train = pd.DataFrame([0] * n_img + [1] * n_img + [2] * n_img + [3] * n_img + [4] * n_img + [5] * n_img)
train



# loading training images
train_img = []
for i in range(0, 6):
    # defining the image path
    image_path = 'intel-image-classification/seg_test/seg_test/%s/'%traindict[i] # Jay/Jay/%d.jpg'%i

    files = listdir(image_path)

    for f in files[0:n_img]:
        path = image_path + f
        # reading the image
        img = imread(path)
        # normalizing the pixel values
        img = img/255
        # resizing the image to (224,224,3)
        img = resize(img, output_shape=(224,224,3), mode='constant', anti_aliasing=True)
        # converting the type of pixel to float 32
        img = img.astype('float32')
        # appending the image into the list
        train_img.append(img)

# converting the list to numpy array
train_x = np.array(train_img)
train_y = train[0].values
train_x.shape, train_y.shape



print(train_y[125])
plt.imshow(train_x[125])



# create validation set
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.2, random_state = 13, stratify=train_y)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.3, random_state = 13, stratify=train_y)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape), (test_x.shape, test_y.shape)



# settings of data augmentation for train_x & val_x
# train
train_rgb_mean = list(train_x.mean(axis = (0,1,2)))
train_rgb_std = list(train_x.std(axis = (0,1,2)))
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees = (30,60)),
    transforms.ColorJitter(brightness=10, contrast=10, saturation=10), # Randomly change the brightness, contrast and saturation of an image
    transforms.ToTensor(),
    transforms.Normalize(train_rgb_mean, train_rgb_std),
])
print(train_rgb_mean, train_rgb_std)

# valid, settings should be same as training
transform_valid = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees = (30,60)),
    transforms.ColorJitter(brightness=10, contrast=10, saturation=10), # Randomly change the brightness, contrast and saturation of an image
    transforms.ToTensor(),
    transforms.Normalize(train_rgb_mean, train_rgb_std), # using train_rgb_mean、train_rgb_std, same as training
])



# converting training images into torch format
train_x = train_x.reshape(train_x.shape[0], train_x.shape[3], train_x.shape[1], train_x.shape[2])
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# shape of training data
train_x.shape, train_y.shape



# converting validation images into torch format
val_x = val_x.reshape(val_x.shape[0], val_x.shape[3], val_x.shape[1], val_x.shape[2])
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)

# shape of validation data
val_x.shape, val_y.shape



# converting testing images into torch format
test_x = test_x.reshape(test_x.shape[0], test_x.shape[3], test_x.shape[1], test_x.shape[2])
test_x  = torch.from_numpy(test_x)

# converting the target into torch format
test_y = test_y.astype(int)
test_y = torch.from_numpy(test_y)

# shape of training data
test_x.shape, test_y.shape



# do data augmentation
# transformed_train_x = torch.zeros(train_x.size())
# transformed_val_x = torch.zeros(val_x.size())

# train
# print(train_x[0])
for i in range(len(train_x)):
#     transformed_train_x[i] = transform_train(train_x[i])
    train_x = torch.cat((train_x, transform_train(train_x[i]).unsqueeze(0)), 0) # append transformed images into old train

train_y = torch.cat((train_y,train_y), 0) # double train_y
# print(train_x[0])
# print(train_x[120])
print(train_x.size(), train_y.size())


# valid
for i in range(len(val_x)):
    val_x = torch.cat((val_x, transform_valid(val_x[i]).unsqueeze(0)), 0) # append transformed images into old valid

val_y = torch.cat((val_y,val_y), 0) # double val_y
# print(train_x[0])
# print(train_x[120])
print(val_x.size(), val_y.size())

 [markdown]
# # ResNet18


# loading the pretrained model
model = models.resnet18(pretrained=True)
# model = resnet18(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False
print(model)



# Reset the final fc layer
fc = model.fc
num_ftrs = model.fc.out_features
feature_size = 64
classes_num = 6


model.fc = Sequential(
    fc,
    BatchNorm1d(num_ftrs),
#     Dropout(0.5),
    Linear(num_ftrs, feature_size),
    BatchNorm1d(feature_size),
    ELU(inplace=True),
#     Dropout(0.5),
    Linear(feature_size, classes_num),
)

# train fc
for param in model.fc.parameters():
    param.requires_grad = True

for param in model.layer4[1].conv2.parameters():
    param.requires_grad = True

# set different learning rate

# model exclude fc
ignored_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda x: id(x) not in ignored_params, model.parameters())
params_list = [{'params': base_params, 'lr': 0.0001},]
params_list.append({'params': model.fc.parameters(), 'lr': 0.0001})
# print(model)



# defining the optimizer
# optimizer = Adam(filter(lambda x : x.requires_grad, model.parameters()), lr=0.0001) # momentum = 0.9
optimizer = Adam(params_list, lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


# Training the model


def fit(epoch, model, data_x, data_y, batch_size = 8, phase = 'training', volatile = False):
    if phase == 'training':
        model.train()
    elif phase == 'validation':
        model.eval()
        volatile = True

    loss, correct = 0.0, 0
    losses = []

    # Returns a random permutation of integers from 0 to n - 1.
    permutation = torch.randperm(data_x.size()[0])
    for i in tqdm(range(0,data_x.size()[0], batch_size)):

        indices = permutation[i : i + batch_size]
        batch_x, batch_y = data_x[indices], data_y[indices]

        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        if phase == 'training':
            optimizer.zero_grad()

        outputs = model(batch_x)
        batch_y = batch_y.long() # 必須轉long
        loss = criterion(outputs, batch_y)

        preds = outputs.data.max(dim = 1, keepdim = True)[1]
        correct += preds.eq(batch_y.data.view_as(preds)).cpu().sum().numpy()

        if phase == 'training':
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    loss = np.average(losses)
    accuracy = correct / len(data_y)
    print('epoch: {}, {} loss is {:8.4f} and {} accuracy is {:8.4f}'.format(epoch, phase,
                                                                    loss, phase, accuracy))
    return loss, accuracy


# batch size of the model
batch_size = 32
# number of epochs to train the model
epoches = 5

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
start = time.time()
for epoch in range(0, epoches):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_x, train_y, batch_size, phase = 'training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, val_x, val_y, batch_size, phase = 'validation')
    print('=' * 10)
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
end = time.time()
print("Training time: " + str(end - start))


### 觀察loss & accuracy
plt.figure(1)
plt.plot(train_losses, 'bo', label = 'training loss')
plt.plot(val_losses, 'r', label = 'validation loss')
plt.title("Loss")
plt.legend()

plt.figure(2)
plt.plot(train_accuracy, 'bo', label = 'training accuracy')
plt.plot(val_accuracy, 'r', label = 'validation accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()


# prediction for tetsing set
prediction_test = []
target_test = []
permutation = torch.randperm(test_x.size()[0])
correct = 0
for i in tqdm(range(0, test_x.size()[0], batch_size)):
    indices = permutation[i : i + batch_size]
    batch_x, batch_y = test_x[indices], test_y[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    ###
    batch_y = batch_y.long() # 必須轉long
    preds = output.data.max(dim = 1, keepdim = True)[1]
    correct += preds.eq(batch_y.data.view_as(preds)).cpu().sum().numpy()
    prediction_test.append(preds)
    target_test.append(batch_y)

for i in range(len(target_test)):
    print(target_test[i])
    print(prediction_test[i].view(-1))
    print("--" * 10)
print()
print('testing accuracy: ', (correct/len(test_y)))


### 預測結果
print("Real value", batch_y[-1])
print("Predicted value", preds[-1])
plt.imshow(batch_x[-1].reshape(224, 224, 3))


# loading training images
test_img = []
image_path =  'chart/testdata/'
files = listdir(image_path)

for f in files[0:n_img]:
    path = image_path + f
    # reading the image
    img = imread(path)
    # normalizing the pixel values
    img = img/255
    # resizing the image to (224,224,3)
    img = resize(img, output_shape=(224,224,3), mode='constant', anti_aliasing=True)
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    test_img.append(img)

# converting the list to numpy array
test_img = np.array(test_img)

# converting testing images into torch format
test_img = test_img.reshape(test_img.shape[0], test_img.shape[3], test_img.shape[1], test_img.shape[2])
test_img  = torch.from_numpy(test_img)

# shape of training data
test_img.shape


# prediction for testing set
if torch.cuda.is_available():
    test_img = test_img.cuda()

with torch.no_grad():
    pred_img = model(test_img)

preds_img = pred_img.data.max(dim = 1, keepdim = True)[1]
for i in range(len(preds_img)):
    print("class ", int(preds_img[i]), " : ", traindict[int(preds_img[i])])
    plt.imshow(test_img[i].reshape(224, 224, 3))
    plt.axis('off')
    plt.show()
    print("==" * 10)
