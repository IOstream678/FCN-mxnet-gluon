import sys

sys.path.append('../')
from mxnet import gluon
import d2lzh as d2l
from FCN.utils import ctx, get_VOC_train_iter, get_VOC_test_iter, loss, \
    num_epochs, get_net_FCN_32s

net_FCN_32s = get_net_FCN_32s(ctx)
net_FCN_32s.hybridize()
# print(net_FCN_32s(X).shape)


trainer = gluon.Trainer(net_FCN_32s.collect_params(), 'sgd', {'learning_rate': 0.1,
                                                              'wd': 1e-5})
train_iter = get_VOC_train_iter()
test_iter = get_VOC_test_iter()
d2l.train(train_iter, test_iter, net_FCN_32s, loss, trainer, ctx, num_epochs)
net_FCN_32s.save_parameters(filename='vgg16_FCN-32s.params')
