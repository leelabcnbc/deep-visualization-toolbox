#! /usr/bin/env python
# Make sure that caffe is on the python path:

import caffe
import numpy as np
import os.path

caffe_root = os.path.normpath(os.path.abspath(os.path.join(caffe.__path__[0], '..', '..')))


def load_imagenet_mean():
    imagenet_mean = np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1)
    assert imagenet_mean.shape == (3,)
    # I will simply use the cross-patch mean, as this is more portable, and should not make a difference
    # for visualization purpose.
    return imagenet_mean
