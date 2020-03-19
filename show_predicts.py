import mxnet
from mxnet.gluon import nn
from mxnet import nd
import d2lzh as d2l
import pylab
from FCN.utils import get_vgg16_features_extractor, get_VOC_train_iter, get_VOC_test_iter, num_classes, \
    label2image, batch_size, get_net_FCN_32s

predict_ctx = [mxnet.gpu(0)]
net_FCN_32s = get_net_FCN_32s(predict_ctx)
net_FCN_32s.hybridize()
# print(net_FCN_32s)

train_or_test_iter = get_VOC_train_iter()

images = []
for X, y in train_or_test_iter:
    X = X.as_in_context(predict_ctx[0])
    y = y.as_in_context(predict_ctx[0])
    n = X.shape[0]
    y_hat = nd.argmax(net_FCN_32s(X), axis=1)
    print(y_hat)
    for i in range(n):
        x = X[i, :, :, :].transpose((1, 2, 0))
        pred_image = label2image(y_hat[i, :, :])
        true_image = label2image(y[i, :, :])
        images += [x, pred_image, true_image]
    break

d2l.show_images(images[::3] + images[1::3], images[2::3], 3, batch_size)
pylab.show()
