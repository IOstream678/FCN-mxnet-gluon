# FCN-mxnet-gluon
本项目是Fully convolutional networks for semantic segmentation论文的复现，论文地址：https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
<br />本项目参考自《动手学深度学习》FCN章节，地址：https://zh.d2l.ai/chapter_computer-vision/fcn.html

需要提前准备好数据集PASCAL VOC 2012 使用方法见教程：https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html

需要将d2lzh工具包解压放置在../目录下

只需依次运行FCN-32s.py， FCN-16s.py, FCN-8s.py即可<br/>show_predicts.py用于查看网络在数据集上的预测结果，可以按需自行修改
