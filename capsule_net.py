import mxnet as mx
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from capsule_block import CapConvBlock, CapFullyBlock, LengthBlock
from conv_cap import AdvConvCap, PrimeConvCap




def CapsNet(batch_size, ctx):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=256, kernel_size=9, strides=1,
                              padding=(0,0), activation='relu'))
        net.add(PrimeConvCap(32,8, kernel_size=(9,9), strides=(2,2), padding=(0,0)))
        # net.add(AdvConvCap(32,8,32,8,3))
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
