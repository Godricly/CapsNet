import sys
import os
import argparse
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
        net.initialize(ctx=ctx, init=init.Xavier())
    return net

def ClsNet(batch_size, ctx):
    net = nn.Sequential()
    with net.name_scope():
        net.add(LengthBlock())
    net.initialize(ctx=ctx, init=init.Xavier())
    return net

def CapLoss(y_pred, y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))

def ReconNet(batch_size, ctx):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(512, activation='relu'))
        net.add(nn.Dense(1024, activation='relu'))
        net.add(nn.Dense(784, activation='sigmoid'))
        net.initialize(ctx=ctx, init=init.Xavier())
    return net

def RecLoss(rec_x, x):
    x_reshape = x.reshape((0, -1))
    diff = nd.square(x_reshape - rec_x)
    return nd.mean(nd.sum(diff, axis=1))


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--recon', action='store_true')
    args = parser.parse_args()
    print(args)

    print_batches = 250 
    ctx = mx.gpu(0)
    train_data, test_data = utils.load_data_mnist(batch_size=args.batch_size,resize=28)

    capnet = CapsNet(args.batch_size, ctx)
    clsnet = ClsNet(args.batch_size, ctx)
    captrainer = Trainer(capnet.collect_params(),'adam', {'learning_rate': 0.001})
    if args.recon:
        recnet = ReconNet(args.batch_size, ctx)
        rectrainer = Trainer(recnet.collect_params(),'adam', {'learning_rate': 0.001})
        
        utils.train_caprec(train_data, test_data, capnet, clsnet, recnet, CapLoss, RecLoss,
                    captrainer, rectrainer, ctx,
                    num_epochs=args.epochs, print_batches=print_batches)
    else:
        utils.train(train_data, test_data, capnet, clsnet,
                    CapLoss, captrainer, ctx,
                    num_epochs=args.epochs, print_batches=print_batches)
