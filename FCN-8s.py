import sys

sys.path.append('../')
from mxnet import gluon, nd
import d2lzh as d2l
from FCN.utils import ctx, get_VOC_train_iter, loss, \
    num_epochs, get_VOC_test_iter, get_net_FCN_8s

net_FCN_8s = get_net_FCN_8s(ctx, FCN_16s_params_file='vgg16_FCN-16s.params')

print(net_FCN_8s(nd.random.normal(shape=(1, 3, 320, 480), ctx=ctx[0])).shape)
net_FCN_8s.hybridize()
trainer = gluon.Trainer(net_FCN_8s.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': 0.9,
                                                              'wd': 5e-4})
train_iter = get_VOC_train_iter()
test_iter = get_VOC_test_iter()
d2l.train(train_iter, test_iter, net_FCN_8s, loss, trainer, ctx, num_epochs)
net_FCN_8s.save_parameters(filename='vgg16_FCN-16s.params')
