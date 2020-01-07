"""InceptionResnetV2 Default model from Tensorflow 2.0."""
from tensorflow.keras import applications
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from libs.pick_random_sample import sampling
from libs.pre_processing import preprocess
from libs.post_processing import postprocess
import math

WIDTH = 299
HEIGHT = 299


def training(batch_size: int, lr, opt, loss_func, sample_size=100):
    """Train DataSet Generator with Augmentation."""
    print("\nTraining Data Set")
    train_flow = preprocess(sampling("train", sample_size), preprocess_input,
                            HEIGHT, WIDTH, batch_size)
    # Test DataSet Generator with Augmentation
    print("\nTest Data Set")
    test_flow = preprocess(sampling("test", sample_size), preprocess_input,
                           HEIGHT, WIDTH, batch_size)

    # Loading the InceptionResNetV2 model and adjusting last layers

    base_model = applications.InceptionResNetV2(weights='imagenet',
                                                include_top=False,
                                                input_shape=(WIDTH, HEIGHT, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(len(train_flow.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=opt(lr=lr),
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
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
                        epochs=15,
                        verbose=1,
                        steps_per_epoch=math.ceil(
                            train_flow.samples/train_flow.batch_size),
                        callbacks=[checkpoint, early],
                        validation_data=(test_flow))
    postprocess(history, "inception_v4", "accuracy")
    postprocess(history, "inception_v4", "loss")
    # Save the model
    model.save('../inception_v4.hdf5')
