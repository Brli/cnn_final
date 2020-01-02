from matplotlib import pyplot as plt


def postprocess(history, name):
    # Plot training accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'], 'rb')
    plt.title(name + 'Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("../" + name + "_accuracy.png")

    # Plot training loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'], 'rb')
    plt.title(name + 'Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("../" + name + "_loss.png")
