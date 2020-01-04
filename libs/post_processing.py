from matplotlib import pyplot as plt


def postprocess(history, name: str, plot_type: str):
    # Plot training accuracy values
    # history: History object
    # name: name of algorithm used
    # plot_type: type of plot wanted, ie. accuracy or loss

    epochs = range(1, len(history.epoch) + 1)
    val_plot_type_object = history.history['val_' + plot_type]
    plot_type_object = history.history[plot_type]
    last_id = len(plot_type_object) - 1
    # epochs: number of ephoch, use length of history.epoch begin with 0
    #         check for python range usage for the "+1"
    # fetch one of the the history.history['key'] list object
    # val_plot_type_object: validation test part
    # plot_type_object: trainning accuracy part
    # last_id: used to plot the last statistics, which should be the most accurate
    # FIXME: try to point out the "best" instead of the "last"
    label = "({}, {:.2f})".format(len(history.epoch),
                                  plot_type_object[last_id])
    plt.annotate(label,  # this is the text
                 xy=(len(history.epoch),
                     plot_type_object[last_id]),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center',
                 va='bottom')
    plt.plot(epochs, plot_type_object, 'r')
    plt.plot(epochs, val_plot_type_object, 'bo')
    plt.title(name + ' Model ' + plot_type)
    plt.ylabel(plot_type.capitalize())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("../" + name + "_" + plot_type + ".png")
    # need to clear the figure or the next figure would have overlapped curve
    plt.clf()
