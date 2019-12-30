from matplotlib import pyplot as plt

def postprocess(history):
    # Plot training accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
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
