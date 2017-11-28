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
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--recon', action='store_true')
    args = parser.parse_args()
    print(args)

    ctx = mx.gpu(0)
    capnet = CapsNet(args.batch_size, ctx)
    capnet.load_params('capnet.params', ctx)
    train_data, test_data = utils.load_data_mnist(batch_size=args.batch_size,resize=28)
    for batch in test_data:
        data, label = utils._get_batch(batch, ctx)
        one_hot_label = nd.one_hot(label,10)
        capout = capnet(data)
        masked_capoutput = capout * nd.expand_dims(one_hot_label, axis=1)
        input_x = vis_square(data.asnumpy().reshape(-1,28,28))
        cv2.imwrite('rec/input_x.png', (input_x*255).astype(int))
        if args.recon:
           recnet = ReconNet(args.batch_size, ctx)
           recnet.load_params('recnet.params', ctx)
           recoutput = recnet(masked_capoutput)
           rec_x = vis_square(recoutput.asnumpy().reshape(-1,28,28))
           cv2.imwrite('rec/rec_x.png', (rec_x*255).astype(int))
        break
