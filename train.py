import sys
import os
import argparse
sys.path.insert(0,'../incubator-mxnet/python')
import mxnet as mx
import utils
from mxnet.gluon import Trainer
from capsule_net import CapsNet, ClsNet, ReconNet, CapLoss, RecLoss

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--recon', action='store_true')
    args = parser.parse_args()
    print(args)

    print_batches = 50 
    ctx = mx.gpu(1)
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

    capnet.save_params('capnet.params')
    if args.recon:
        recnet.save_params('recnet.params')
