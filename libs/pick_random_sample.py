from os import isdir, makedirs, rmtree, listdir
from os.path import join
from shutil import copy
import random

def __init__(self, act: str, sample_size: int):
    # act should be stricted to "train" or "test"
    # sample size should be tunable out of scope

    # Printing the name of the classes of the data
    src_dir = join("data", act)

    classes = listdir(src_dir)
    print("The Classes of images are as follows: ")
    print(', '.join(classes))

    # Pick random sample for training input
    # the project utilize directory structure be like seg_train/<item class name>/XXX.{image mimetype}
    # data/ stores the preset intel_image_classification images
    # samples/ stores the following sampled images
    img_number = sample_size
    for cls in classes:
        target_dir = join("samples", act)
        if isdir(target_dir):
        # Clear the exist directory before we start sampling
            rmtree(target_dir)
        # We pick the "img_number" amount of samples
        # NOTE: that the number may be less, since we didn't deal with redundency
        makedirs(target_dir)
        for x in range(0, img_number):
            samples = random.choice(listdir(join(src_dir, cls)))
            copy(join(src_dir, cls, samples), join(target_dir, cls, samples))
