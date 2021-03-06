"""Pick samples from dataset for training."""
from os import makedirs, listdir
from os.path import join, isdir
from shutil import copy, rmtree
import random


def sampling(act: str, sample_size=50):
    """
    Pick random sample for training input.

    Parameters:
        act: only accept "train" or "test"
        sample_size: define the quantity of images selected, default to 50.
    """
    if act in ["train", "test"]:
        pass
    else:
        raise ValueError("Action should be either train or test")

    src_dir = join("data", act)
    classes = listdir(src_dir)
    print("The Classes of images are as follows: ")
    print(', '.join(classes))

    # the project utilize directory structure is like:
    #   <root_dir>/<act>/<item class name>/XXX.{image mimetype}
    # root_dir is 'data' or 'samples'
    #   data/ stores the preset intel_image_classification images
    #   samples/ stores the following sampled images
    # act is train or test
    # class name is the name of image set
    target_dir = join("samples", act)
    if isdir(target_dir):
        # Clear the exist directory before we start sampling
        rmtree(target_dir)

    for cls in classes:
        # We pick the "img_number" amount of samples
        # NOTE: the number may be less, since we didn't deal with redundancy
        makedirs(join(target_dir, cls))
        for x in range(0, sample_size):
            samples = random.choice(listdir(join(src_dir, cls)))
            copy(join(src_dir, cls, samples), join(target_dir, cls, samples))

    return target_dir
