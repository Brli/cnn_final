#!/usr/bin/env python
# coding: utf-8

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
from torch.backends import cudnn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models

traindict = {0 : 'buildings', 1 : 'forest', 2 : 'glacier', 3 : 'mountain', 4 : 'sea', 5 : 'street'}
traindict

train = pd.DataFrame([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25 + [4] * 25 + [5] * 25)
train

# loading training images
train_img = []
for i in range(0, 6):
    # defining the image path
    image_path = 'data/train/%s/'%traindict[i] # Jay/Jay/%d.jpg'%i

    files = listdir(image_path)

    for f in files[0:25]:
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
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2, random_state = 13, stratify=train_y)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

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

# # Vgg16
# loading the pretrained model
model = models.vgg16_bn(pretrained=True)
# Freeze model weights
for param in model.parameters():
    param.requires_grad = False
print(model)

# Add on classifier
model.classifier[6] = Sequential(
                      Linear(4096, 6))
for param in model.classifier[6].parameters():
    param.requires_grad = True

# defining the optimizer
# torch.cuda.empty_cache()
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# print(model)

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# print(model)

# batch size of the model
batch_size = 8

# number of epochs to train the model
epochs = 15

for epoch in range(0, epochs):

    # keep track of training and validation loss
    train_loss = 0.0

    # Returns a random permutation of integers from 0 to n - 1.
    permutation = torch.randperm(train_x.size()[0]).cuda()

    training_loss = []
    for i in tqdm(range(0,train_x.size()[0], batch_size)):

        indices = permutation[i : i + batch_size]
        batch_x, batch_y = train_x[indices], train_y[indices]

        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        # in case you wanted a semi-full example
        outputs = model(batch_x)
        batch_y = batch_y.long() # 必須轉long
        loss = criterion(outputs, batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)

# prediction for training set
prediction = []
target = []
permutation = torch.randperm(train_x.size()[0])
for i in tqdm(range(0,train_x.size()[0], batch_size)):
    indices = permutation[i : i + batch_size]
    batch_x, batch_y = train_x[indices], train_y[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)

# training accuracy
accuracy = []
for i in range(len(prediction)):
    print(target[i],prediction[i])
    accuracy.append(accuracy_score(target[i].cpu(), prediction[i]))

print('training accuracy: \t', np.average(accuracy))

# prediction for validation set
prediction_val = []
target_val = []
permutation = torch.randperm(val_x.size()[0])
for i in tqdm(range(0, val_x.size()[0], batch_size)):
    indices = permutation[i : i + batch_size]
    batch_x, batch_y = val_x[indices], val_y[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_val.append(predictions)
    target_val.append(batch_y)

# validation accuracy
accuracy_val = []
for i in range(len(prediction_val)):
    print(target_val[i],prediction_val[i])
    accuracy_val.append(accuracy_score(target_val[i].cpu(),prediction_val[i]))

print('validation accuracy: \t', np.average(accuracy_val))

torch.cuda.empty_cache()
