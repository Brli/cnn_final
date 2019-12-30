from os import makedirs, listdir
from os.path import join, isdir
from shutil import copy, rmtree
import random


def __init__(self, act: str, sample_size=50):
    # @act only accept "train" or "test"
    # @sample_size should be tunable out of scope, default to 50
    if act in ["train", "test"]:
        pass
    else:
        raise ValueError("Action should be either train or test")

    src_dir = join("data", act)
    classes = listdir(src_dir)
    # sample size should be tunable out of scope
    print("The Classes of images are as follows: ")
    print(', '.join(classes))

    # Pick random sample for training input
    # the project utilize directory structure is like:
    #   <root_dir>/seg_train/<item class name>/XXX.{image mimetype}
    # root_dir is 'data' or 'samples'
    # data/ stores the preset intel_image_classification images
    # samples/ stores the following sampled images
    img_number = sample_size
    for cls in classes:
        target_dir = join("samples", act)
        if isdir(target_dir):
            # Clear the exist directory before we start sampling
            rmtree(target_dir)
        # We pick the "img_number" amount of samples
        # NOTE: the number may be less, since we didn't deal with redundency
        makedirs(target_dir)
        for x in range(0, img_number):
            samples = random.choice(listdir(join(src_dir, cls)))
            copy(join(src_dir, cls, samples), join(target_dir, cls, samples))
