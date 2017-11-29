import sys
import os
import argparse
sys.path.insert(0,'../incubator-mxnet/python')
import mxnet as mx
import utils
import cv2
import numpy as np
from capsule_net import CapsNet, ClsNet, ReconNet, CapLoss, RecLoss
from mxnet import nd


# Sorry, I can't remember who wrote this display code, please contact me for credit ;)
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    # data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=0)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data
    # plt.imshow(data); plt.axis('off')
    



if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()
    print(args)

    ctx = mx.gpu(0)
    capnet = CapsNet(args.batch_size, ctx)
    capnet.load_params('capnet.params', ctx)
    recnet = ReconNet(args.batch_size, ctx)
    recnet.load_params('recnet.params', ctx)

    train_data, test_data = utils.load_data_mnist(batch_size=args.batch_size,resize=28)
    sum_capout = mx.nd.zeros((16, 10), ctx)
    sum_label = mx.nd.zeros((10), ctx)
    for i, batch in enumerate(test_data):
        data, label = utils._get_batch(batch, ctx)
        one_hot_label = nd.one_hot(label, 10)
        capout = capnet(data)
        # maybe I should not use label to create mask
        masked_capoutput = capout * nd.expand_dims(one_hot_label, axis=1)
        sum_capout += nd.sum(masked_capoutput, axis=0)
        sum_label += nd.sum(one_hot_label, axis=0)

        recoutput = recnet(masked_capoutput)
        rec_x = vis_square(recoutput.asnumpy().reshape(-1,28,28))
        # uncomment to plot rec
        if i == 0:
            input_x = vis_square(data.asnumpy().reshape(-1,28,28))
            # cv2.imwrite('rec/input_x.png', (input_x*255).astype(int))
            # cv2.imwrite('rec/rec_x.png', (rec_x*255).astype(int))
    mean_capout = sum_capout / sum_label
    mask = nd.one_hot(mx.nd.array(range(10), ctx), 10)

    cap_mean = mean_capout.reshape((1, 16, 10)).broadcast_to((args.batch_size, 16, 10))
    n = int(np.ceil(np.sqrt(args.batch_size)))
    digit_label = (np.arange(args.batch_size) / n) % 10
    digit_mask = nd.one_hot(mx.nd.array(digit_label, ctx), 10)

    channel_label = (np.arange(args.batch_size) % n) % 16
    channel_mask = nd.one_hot(mx.nd.array(channel_label, ctx), 16)
    channel_mask = channel_mask.reshape((args.batch_size,16,1)).broadcast_to((args.batch_size, 16, 10))

    '''
    # each row in display correspond to same dim
    dim_mask = nd.one_hot(mx.nd.array(range(16), ctx), 16)
    

    print dim_mask

    '''  
    for i, v in enumerate(np.arange(-0.25, 0.3, 0.05)):
        cap_mod = (cap_mean + v* channel_mask) * nd.expand_dims(digit_mask, axis=1)
        recoutput = recnet(cap_mod)
        rec_x = vis_square(recoutput.asnumpy().reshape(-1,28,28))
        cv2.imwrite('rec/rec_x_' + str(i) + '.png', (rec_x*255).astype(int))
