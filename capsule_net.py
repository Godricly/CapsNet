import sys
import os
sys.path.insert(0,'../incubator-mxnet/python')

import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn,Trainer

from capsule_block import CapConvBlock, CapFullyBlock, LengthBlock
import utils




def CapsNet(batch_size, ctx):

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=256, kernel_size=9, strides=1,
                          padding=(0,0), activation='relu'))
        net.add(CapConvBlock(32, 8, kernel_size=(9, 9),
                            strides=(2, 2), padding=(0,0), context=ctx))
        net.add(CapFullyBlock(36*32,10,8,16,context=ctx))
        net.add(LengthBlock())
    net.initialize(ctx=ctx, init=init.Xavier())
    return net


def loss(y_pred,y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train', default=False, type=bool)
    args = parser.parse_args()
    print(args)

    ctx = mx.gpu(0)
    train_data, test_data = utils.load_data_mnist(batch_size=args.batch_size,resize=28)
    net = CapsNet(32, ctx)
    print('====================================net====================================')
    trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.01})
    print_batches = 250 
    utils.train(train_data, test_data, net, loss, trainer, ctx,
                num_epochs=args.epochs, print_batches=print_batches)
