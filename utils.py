import os
import sys

import mxnet, d2lzh as d2l
import numpy as np
from mxnet import nd, init
from mxnet.gluon import model_zoo, data as gdata, loss as gloss, nn

num_classes = 21
batch_size = 16
ctx = [mxnet.gpu(0)]
num_epochs = 400
crop_size = (320, 480)


def get_vgg16_features_extractor(ctx):
    vgg16net = model_zoo.vision.get_model('vgg16', pretrained=True, ctx=ctx)
    features_extractor = vgg16net.features[:31]
    return features_extractor


def get_VOC_train_iter():
    colormap2label = nd.zeros(256 ** 3)
    for i, cm in enumerate(d2l.VOC_COLORMAP):
        colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
    data_dir = '../data'
    voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        d2l.VOCSegDataset(True, crop_size, voc_dir, colormap2label), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    return train_iter


def get_VOC_test_iter():
    colormap2label = nd.zeros(256 ** 3)
    for i, cm in enumerate(d2l.VOC_COLORMAP):
        colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
    data_dir = '../data'
    voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
    num_workers = 0 if sys.platform.startswith('win32') else 4
    test_iter = gdata.DataLoader(
        d2l.VOCSegDataset(False, crop_size, voc_dir, colormap2label), batch_size,
        last_batch='discard', num_workers=num_workers)
    return test_iter


loss = gloss.SoftmaxCrossEntropyLoss(axis=1)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)


def label2image(pred):
    colormap = nd.array(d2l.VOC_COLORMAP, ctx=ctx[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]


def get_net_FCN_32s(ctx):
    features_extractor = get_vgg16_features_extractor(ctx)
    # 构建FCN-32s网络
    net_FCN_32s = nn.HybridSequential()
    for layer in features_extractor:
        net_FCN_32s.add(layer)
    net_FCN_32s.add(nn.Conv2D(channels=num_classes, kernel_size=1),
                    nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16,  # 感觉这个上采样有点问题，是精度不行的元凶
                                       strides=32)
                    )
    net_FCN_32s[-2].initialize(mxnet.init.Xavier(), ctx=ctx)
    net_FCN_32s[-1].initialize(mxnet.init.Constant(bilinear_kernel(num_classes, num_classes, 64)), ctx=ctx)
    return net_FCN_32s


def get_net_FCN_16s(ctx):
    class FCN_16s(nn.HybridBlock):
        def __init__(self, ctx):
            super(FCN_16s, self).__init__()
            with self.name_scope():
                self.feature_extractor = get_vgg16_features_extractor(ctx)
                self.feature_extractor.add(nn.Conv2D(channels=num_classes, kernel_size=1),
                                           nn.Conv2DTranspose(num_classes, in_channels=num_classes,kernel_size=4, padding=1,
                                                              strides=2))  # 2x上采样
                self.feature_extractor[-2].initialize(init.Xavier(), ctx=ctx)
                self.feature_extractor[-1].initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 4)),
                                                      ctx=ctx)
                self.conv1x1_pool4 = nn.Conv2D(channels=num_classes, kernel_size=1)
                self.conv1x1_pool4.initialize(init.Zero(), ctx=ctx)
                self.convTrans_16x = nn.Conv2DTranspose(num_classes, in_channels=num_classes, kernel_size=32, padding=8, strides=16)  # 16x上采样
                self.convTrans_16x.initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 32)), ctx=ctx)

        def hybrid_forward(self, F, x, *args, **kwargs):
            x_pool4 = self.feature_extractor[:24](x)
            # print(x_pool4.shape)
            x_conv7 = self.feature_extractor[24:](x_pool4)
            # print(x_conv7.shape)
            x_pool4 = self.conv1x1_pool4(x_pool4)
            x_sum = x_pool4 + x_conv7
            return self.convTrans_16x(x_sum)

    return FCN_16s(ctx)


def get_net_FCN_8s(ctx, FCN_16s_params_file):
    class FCN_8s(nn.HybridBlock):
        def __init__(self, ctx, FCN_16s_params_file):
            super(FCN_8s, self).__init__()
            with self.name_scope():
                self.FCN_16s = get_net_FCN_16s(ctx)
                self.FCN_16s.load_parameters(filename=FCN_16s_params_file, ctx=ctx)
                # self.FCN_16s.hybridize()
                self.conv1x1_pool3 = nn.Conv2D(channels=num_classes, kernel_size=1)
                self.conv1x1_pool3.initialize(init.Zero(), ctx=ctx)
                self.convTrans_2x = nn.Conv2DTranspose(num_classes, kernel_size=4, padding=1, strides=2)   # 2x上采样
                self.convTrans_2x.initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 4)), ctx=ctx)
                self.convTrans_4x = nn.Conv2DTranspose(num_classes, kernel_size=8, padding=2, strides=4)   # 4x上采样
                self.convTrans_4x.initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 8)), ctx=ctx)
                self.convTrans_8x = nn.Conv2DTranspose(num_classes, kernel_size=16, padding=4, strides=8)  # 8x上采样
                self.convTrans_8x.initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 16)), ctx=ctx)
                # self.convTrans_16x.initialize(init.Constant(bilinear_kernel(num_classes, num_classes, 32)), ctx=ctx)

        def hybrid_forward(self, F, x, *args, **kwargs):
            x_pool3 = self.FCN_16s.feature_extractor[:17](x)

            x_pool4 = self.FCN_16s.feature_extractor[17:24](x_pool3)

            x_conv7 = self.FCN_16s.feature_extractor[24:-1](x_pool4)
            x_pool3 = self.conv1x1_pool3(x_pool3)
            x_pool4 = self.FCN_16s.conv1x1_pool4(x_pool4)
            x_pool4 = self.convTrans_2x(x_pool4)
            x_conv7 = self.convTrans_4x(x_conv7)
            x_sum = x_pool3 + x_pool4 + x_conv7
            return self.convTrans_8x(x_sum)

    return FCN_8s(ctx, FCN_16s_params_file)