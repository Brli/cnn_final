from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from libs.pick_random_sample import sampling
import math


WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 8

train_dir = sampling("train")
test_dir = sampling("test")

# Train DataSet Generator with Augmentation
print("\nTraining Data Set")
train_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_flow = train_generator.flow_from_directory(
    train_dir,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE
)

# Test DataSet Generator with Augmentation
print("\nTest Data Set")
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_flow = test_generator.flow_from_directory(
    test_dir,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE
)

# Loading the inceptionV3 model and adjusting last layers

base_model = applications.InceptionV3(weights='imagenet',
                                      include_top=False,
                                      input_shape=(WIDTH, HEIGHT, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(train_flow.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'],
              loss='categorical_crossentropy')
# model.summary()


top_layers_file_path = "model.hdf5"

# Defining the callbacks for the model
checkpoint = ModelCheckpoint(top_layers_file_path,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
early = EarlyStopping(monitor="loss", mode="min", patience=5)

# Training
history = model.fit(train_flow,
                    epochs=20,
                    verbose=1,
                    steps_per_epoch=math.ceil(train_flow.samples/train_flow.batch_size),
                    callbacks=[checkpoint, early],
                    validation_data=(test_flow))

# Plot training accuracy values
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Plot training loss values
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

labels = [k for k, _ in train_flow.class_indices.items()]
file_list = listdir('intel-image-classification/seg_pred')
img_name = random.choice(file_list)
img_path = join('intel-image-classification/seg_pred', img_name)

img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

pred = model.predict(x)
pred_label = labels[np.argmax(pred)]
pred = round(pred[0][np.argmax(pred)] * 100, 2)
if pred >=95:
  print("The label is: ", pred_label)
  print("Confidece: ", pred, "%")
else:
  print("Failed to match")
