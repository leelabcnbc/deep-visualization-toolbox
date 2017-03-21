#! /usr/bin/env python

import argparse
import cPickle as pickle

from loaders import load_imagenet_mean, caffe
from jby_misc import WithTimer
from max_tracker import scan_images_for_maxes


def main():
    parser = argparse.ArgumentParser(
        description='Finds images in a training set that cause max activation for a network; saves results in a pickled NetMaxTracker.')
    parser.add_argument('--N', type=int, default=9, help='note and save top N activations')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    parser.add_argument('net_prototxt', type=str, default='', help='network prototxt to load')
    parser.add_argument('net_weights', type=str, default='', help='network weights to load')
    parser.add_argument('datadir', type=str, default='.', help='directory to look for files in')
    parser.add_argument('filelist', type=str, help='list of image files to consider, one per line')
    parser.add_argument('outfile', type=str, help='output filename for pkl')
    parser.add_argument('net_layers', type=str, help='list of layers to compute, separated by comma')
    parser.add_argument('net_is_conv', type=str,
                        help='list of 1 (True) and 0 (False), separated by comma, same length as net_layers')

    # parser.add_argument('--mean', type = str, default = '', help = 'data mean to load')
    args = parser.parse_args()

    # layers, is_conv
    layers = [x.strip() for x in args.net_layers.split(',')]
    is_conv = [x.strip() for x in args.net_is_conv.split(',')]
    is_conv = [True if x == '1' else False for x in is_conv]
    assert len(layers) == len(is_conv)

    imagenet_mean = load_imagenet_mean()
    # this should be rewritten in transformer + net, and some batch operation should be supported.
    # whatever, just make it usable first.
    # also, this restricts to images with labels, since `filelist` must accept something with label
    # and restrict the dataset to those with size 256x256.
    net = caffe.Classifier(args.net_prototxt, args.net_weights,
                           mean=imagenet_mean,
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(256, 256))
    if args.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    with WithTimer('Scanning images'):
        max_tracker = scan_images_for_maxes(net, args.datadir, args.filelist, args.N,
                                            layers=layers, is_conv=is_conv)
    with WithTimer('Saving maxes'):
        with open(args.outfile, 'wb') as ff:
            pickle.dump(max_tracker, ff, -1)


if __name__ == '__main__':
    main()
