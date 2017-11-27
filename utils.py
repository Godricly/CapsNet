from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
import mxnet as mx

def load_data_fashion_mnist(batch_size, resize=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            data = image.imresize(data, resize, resize)
        # change data from height x weight x channel to channel x height x weight
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(root='./data',
        train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(root='./data',
        train=False, transform=transform_mnist)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)

def load_data_mnist(batch_size, resize=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            data = image.imresize(data, resize, resize)
        # change data from height x weight x channel to channel x height x weight
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = gluon.data.vision.MNIST(root='./data',
        train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.MNIST(root='./data',
        train=False, transform=transform_mnist)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def accuracy(output, label):
    return nd.mean(nd.argmax(output,axis=1)==label).asscalar()

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)


def evaluate_accuracy(data_iterator, capnet, clsnet, ctx=mx.gpu()):
    acc = 0.
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        capoutput = capnet(data)
        clsoutput = clsnet(capoutput)
        acc += accuracy(clsoutput, label)
    return acc / (i+1)


def train(train_data, test_data, capnet, clsnet,
          loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        n = 0
        for i, batch in enumerate(train_data):
            data, label = _get_batch(batch, ctx)
            one_hot_label = nd.one_hot(label,10)
            with autograd.record():
                capoutput = capnet(data)
                clsoutput = clsnet(capoutput)
                L = loss(clsoutput, one_hot_label)
                L.backward()
            trainer.step(data.shape[0],ignore_stale_grad=True)
            train_loss += nd.mean(L).asscalar()
            # print('Loss: ',nd.mean(L).asscalar())
            train_acc += accuracy(clsoutput, label)
            n = i + 1
            if print_batches and n % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                n, train_loss/n, train_acc/n))
        test_acc = evaluate_accuracy(test_data, capnet, clsnet, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
              epoch, train_loss/n, train_acc/n, test_acc))

def train_caprec(train_data, test_data, capnet, clsnet, recnet, ClsLoss, RecLoss,
                 captrainer, rectrainer, ctx, num_epochs, print_batches=None):
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        n = 0
        for i, batch in enumerate(train_data):
            data, label = _get_batch(batch, ctx)
            one_hot_label = nd.one_hot(label,10)
            with autograd.record():
                capoutput = capnet(data)
                masked_capoutput = capoutput * nd.expand_dims(one_hot_label, axis=1)
                clsoutput = clsnet(capoutput)
                clsloss = ClsLoss(clsoutput, one_hot_label)
                recoutput = recnet(masked_capoutput)
                recloss = RecLoss(recoutput, data)
                Loss = clsloss + recloss * 0.0005
                Loss.backward()
            rectrainer.step(data.shape[0],ignore_stale_grad=True)
            captrainer.step(data.shape[0],ignore_stale_grad=True)
            train_loss += nd.mean(Loss).asscalar()
            train_acc += accuracy(clsoutput, label)
            n = i + 1
            if print_batches and n % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                n, train_loss/n, train_acc/n))
        test_acc = evaluate_accuracy(test_data, capnet, clsnet, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
              epoch, train_loss/n, train_acc/n, test_acc))
