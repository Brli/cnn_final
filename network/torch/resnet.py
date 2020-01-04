import pandas as pd
import numpy as np
from tqdm import tqdm

# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, isdir, join

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models

traindict = {0 : 'buildings', 1 : 'forest', 2 : 'glacier', 3 : 'mountain', 4 : 'sea', 5 : 'street'}
n_img = 100
train = pd.DataFrame([0] * n_img + [1] * n_img + [2] * n_img + [3] * n_img + [4] * n_img + [5] * n_img)

# loading training images
train_img = []
for i in range(0, 6):
    # defining the image path
    image_path = '../../data/train/%s/'%traindict[i] # Jay/Jay/%d.jpg'%i

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

# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2, random_state = 13, stratify=train_y)

# converting training images into torch format
train_x = train_x.reshape(train_x.shape[0], train_x.shape[3], train_x.shape[1], train_x.shape[2])
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# converting validation images into torch format
val_x = val_x.reshape(val_x.shape[0], val_x.shape[3], val_x.shape[1], val_x.shape[2])
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)


# loading the pretrained model
model = models.resnet18(pretrained=True)
# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# Reset the final fc layer
num_ftrs = model.fc.in_features
model.fc = Linear(num_ftrs, 6)

# train fc & layer4[1].conv2
for param in model.fc.parameters():
    param.requires_grad = True

for param in model.layer4[1].conv2.parameters():
    param.requires_grad = True


# defining the optimizer
optimizer = Adam(filter(lambda x : x.requires_grad, model.parameters()), lr=1e-4) # momentum = 0.9
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# training the model

def fit(epoch, model, data_x, data_y, batch_size = 32, phase = 'training', volatile = False):
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
    return loss, accuracy

# batch size of the model
batch_size = 32
# number of epochs to train the model
epoches = 20

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(0, epoches):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_x, train_y, batch_size, phase = 'training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, val_x, val_y, batch_size, phase = 'validation')
    print('=' * 10)
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

### 觀察loss & accuracy
plt.figure(1)
plt.plot(train_losses, 'bo', label = 'training loss')
plt.plot(val_losses, 'r', label = 'validation loss')
plt.title("ResNet18 Loss")
plt.legend()
plt.savefig("../1.png")

plt.figure(2)
plt.plot(train_accuracy, 'bo', label = 'training accuracy')
plt.plot(val_accuracy, 'r', label = 'validation accuracy')
plt.title("ResNet18 Accuracy")
plt.legend()
plt.savefig("../2.png")
