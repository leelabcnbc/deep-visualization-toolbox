#! /usr/bin/env python

import skimage.io
import numpy as np


def shownet(net):
    '''Print some stats about a net and its activations'''

    print '%-41s%-31s%s' % ('', 'acts', 'act diffs')
    print '%-45s%-31s%s' % ('', 'params', 'param diffs')
    for k, v in net.blobs.items():
        if k in net.params:
            params = net.params[k]
            for pp, blob in enumerate(params):
                if pp == 0:
                    print '  ', 'P: %-5s' % k,
                else:
                    print ' ' * 11,
                print '%-32s' % repr(blob.data.shape),
                print '%-30s' % ('(%g, %g)' % (blob.data.min(), blob.data.max())),
                print '(%g, %g)' % (blob.diff.min(), blob.diff.max())
        print '%-5s' % k, '%-34s' % repr(v.data.shape),
        print '%-30s' % ('(%g, %g)' % (v.data.min(), v.data.max())),
        print '(%g, %g)' % (v.diff.min(), v.diff.max())


def region_converter(top_slice, bot_size, top_size, filter_width=(1, 1), stride=(1, 1), pad=(0, 0), norm_last=False):
    '''
    Works for conv or pool
    
vector<int> ConvolutionLayer<Dtype>::JBY_region_of_influence(const vector<int>& slice) {
    +  CHECK_EQ(slice.size(), 4) << "slice must have length 4 (ii_start, ii_end, jj_start, jj_end)";
    +  // Crop region to output size
    +  vector<int> sl = vector<int>(slice);
    +  sl[0] = max(0, min(height_out_, slice[0]));
    +  sl[1] = max(0, min(height_out_, slice[1]));
    +  sl[2] = max(0, min(width_out_, slice[2]));
    +  sl[3] = max(0, min(width_out_, slice[3]));
    +  vector<int> roi;
    +  roi.resize(4);
    +  roi[0] = sl[0] * stride_h_ - pad_h_;
    +  roi[1] = (sl[1]-1) * stride_h_ + kernel_h_ - pad_h_;
    +  roi[2] = sl[2] * stride_w_ - pad_w_;
    +  roi[3] = (sl[3]-1) * stride_w_ + kernel_w_ - pad_w_;
    +  return roi;
    +}
    '''
    assert len(top_slice) == 4
    assert len(bot_size) == 2
    assert len(top_size) == 2
    assert len(filter_width) == 2
    assert len(stride) == 2
    assert len(pad) == 2

    # Crop top slice to allowable region
    top_slice = [ss for ss in top_slice]  # Copy list or array -> list

    # top slice[1], topslice[3] are 1 plus the actual index.
    assert top_slice[0] < top_slice[1]
    assert top_slice[2] < top_slice[3]

    top_slice[0] = max(0, min(top_size[0], top_slice[0]))
    top_slice[1] = max(0, min(top_size[0], top_slice[1]))
    top_slice[2] = max(0, min(top_size[1], top_slice[2]))
    top_slice[3] = max(0, min(top_size[1], top_slice[3]))

    bot_slice = [-123] * 4

    bot_slice[0] = top_slice[0] * stride[0] - pad[0];
    bot_slice[1] = (top_slice[1] - 1) * stride[0] + filter_width[0] - pad[0];
    bot_slice[2] = top_slice[2] * stride[1] - pad[1];
    bot_slice[3] = (top_slice[3] - 1) * stride[1] + filter_width[1] - pad[1];

    # I think you should normalize the bottom size as well, sometimes. Otherwise, if there's padding on bottom slice,
    # and this is the last one. then there's no way it will get normalized.

    if norm_last:
        bot_slice[0] = max(0, min(bot_size[0], bot_slice[0]))
        bot_slice[1] = max(0, min(bot_size[0], bot_slice[1]))
        bot_slice[2] = max(0, min(bot_size[1], bot_slice[2]))
        bot_slice[3] = max(0, min(bot_size[1], bot_slice[3]))

    assert bot_slice[0] < bot_slice[1]
    assert bot_slice[2] < bot_slice[3]

    return bot_slice


def get_conv_converter(bot_size, top_size, filter_width=(1, 1), stride=(1, 1), pad=(0, 0)):
    return lambda top_slice, norm_last: region_converter(top_slice, bot_size, top_size, filter_width, stride, pad,
                                                         norm_last)


def get_pool_converter(bot_size, top_size, filter_width=(1, 1), stride=(1, 1), pad=(0, 0)):
    return lambda top_slice, norm_last: region_converter(top_slice, bot_size, top_size, filter_width, stride, pad,
                                                         norm_last)


converter_this_mapping = {'conv': get_conv_converter, 'pool': get_pool_converter}


def get_region_info_one_chain(region_info_list_spec):
    _tmp = []
    assert len(region_info_list_spec) >= 2
    for x in region_info_list_spec:
        name_this = x[0]
        type_this = x[1]
        if type_this is None:
            assert len(x) == 2 or len(x) == 3
            if len(x) == 3:
                assert len(x[2]) == 2 and (x[2][0] is None)
            _tmp.append((name_this, None)),
        else:
            assert len(x) == 3
            args_this = x[2]
            converter_this = converter_this_mapping[type_this](*args_this)
            _tmp.append((name_this, converter_this))
    return _tmp


class RegionComputer(object):
    '''Computes regions of possible influcence from higher layers to lower layers.

    Woefully hardcoded'''

    def __init__(self, region_info_list_specs=None):
        # self.net = net
        if region_info_list_specs is None:
            region_info_list_specs = [
                [
                    ('data', None),
                    ('conv1', 'conv', ((227, 227), (55, 55), (11, 11), (4, 4))),
                    ('pool1', 'pool', ((55, 55), (27, 27), (3, 3), (2, 2))),
                    ('conv2', 'conv', ((27, 27), (27, 27), (5, 5), (1, 1), (2, 2))),
                    ('pool2', 'pool', ((27, 27), (13, 13), (3, 3), (2, 2))),
                    ('conv3', 'conv', ((13, 13), (13, 13), (3, 3), (1, 1), (1, 1))),
                    ('conv4', 'conv', ((13, 13), (13, 13), (3, 3), (1, 1), (1, 1))),
                    ('conv5', 'conv', ((13, 13), (13, 13), (3, 3), (1, 1), (1, 1))),
                ],
            ]




            # # this is the old behavior
            # _tmp = []
            # _tmp.append(('data', None))
            # _tmp.append(('conv1', get_conv_converter((227,227), (55,55), (11,11), (4,4))))
            # _tmp.append(('pool1', get_pool_converter((55,55),   (27,27), (3,3),   (2,2))))
            # _tmp.append(('conv2', get_conv_converter((27,27),   (27,27), (5,5),   (1,1),  (2,2))))
            # _tmp.append(('pool2', get_pool_converter((27,27),   (13,13), (3,3),   (2,2))))
            # _tmp.append(('conv3', get_conv_converter((13,13),   (13,13), (3,3),   (1,1),  (1,1))))
            # _tmp.append(('conv4', get_conv_converter((13,13),   (13,13), (3,3),   (1,1),  (1,1))))
            # _tmp.append(('conv5', get_conv_converter((13,13),   (13,13), (3,3),   (1,1),  (1,1))))
            # self.names = [tt[0] for tt in _tmp]
            # self.converters = [tt[1] for tt in _tmp]

        # then get region_info_list
        region_info_list = [
            get_region_info_one_chain(x) for x in region_info_list_specs
            ]

        self.names_array = [[tt[0] for tt in _tmp] for _tmp in region_info_list]
        self.converters_array = [[tt[1] for tt in _tmp] for _tmp in region_info_list]

        # make sure that names don't share, except possibly for the first one,
        all_names_but_first = []
        len_names_all = 0
        for names in self.names_array:
            len_names_all += len(names[1:])
            all_names_but_first.extend(names[1:])
        assert len(set(all_names_but_first)) == len_names_all

    def determine_converter_chain(self, layer):
        # find the first chain that have this name in it.
        for idx, names in enumerate(self.names_array):
            if layer in names:
                return self.names_array[idx], self.converters_array[idx]

    def convert_region(self, from_layer, to_layer, region, verbose=False, normalize_last=False):
        '''region is the slice of the from_layer in the following Python
            index format: (ii_start, ii_end, jj_start, jj_end)
        '''

        # we should be sure that from_layer only exists in one list of converters.
        names, converters = self.determine_converter_chain(from_layer)

        from_idx = names.index(from_layer)
        to_idx = names.index(to_layer)
        assert from_idx >= to_idx, 'wrong order of from_layer and to_layer'

        ret = region
        for ii in range(from_idx, to_idx, -1):
            converter = converters[ii]
            if verbose:
                print 'pushing', names[ii], 'region', ret, 'through converter'
            ret = converter(ret, normalize_last)
        if verbose:
            print 'Final region at ', names[to_idx], 'is', ret

        return ret


def norm01c(arr, center):
    '''Maps the input range to [0,1] such that the center value maps to .5'''
    arr = arr.copy()
    arr -= center
    arr /= max(2 * arr.max(), -2 * arr.min()) + 1e-10
    arr += .5
    assert arr.min() >= 0
    assert arr.max() <= 1
    return arr


def save_caffe_image(img, filename, autoscale=True, autoscale_center=None):
    '''Takes an image in caffe format (01) or (c01, BGR) and saves it to a file'''
    if len(img.shape) == 2:
        # upsample grayscale 01 -> 01c
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    else:
        img = img[::-1].transpose((1, 2, 0))
    if autoscale_center is not None:
        img = norm01c(img, autoscale_center)
    elif autoscale:
        img = img.copy()
        img -= img.min()
        img *= 1.0 / (img.max() + 1e-10)
    skimage.io.imsave(filename, img)
