"""
Created on Mon Nov 13 16:43:48 2017

@author: Abhijit

Inception-v3 fixed feature architecture + RPN + Classifier for faster RCNN
Adopted from paper "Speed/accuracy trade-offs for modern convolutional object detectors"
Refer Section: 3.6.1 in the paper for more details

NOTE: Feature extracted from "mixed6e"(tensorflow)="mixed7"(keras) whose effective stride = 16
"""
from __future__ import print_function
from __future__ import absolute_import

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
import keras.backend as K
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import TimeDistributed

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              trainable=True):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
        trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name,trainable=trainable)(x)
    x = Activation('relu', name=name,trainable=trainable)(x) #this has no params anyway
    return x


def nn_base(input_tensor=None,trainable=True):
    """ The architecture for fixed feature extractor
    
    # Adopted from "Speed/accuracy trade-offs for modern convolutional object detectors"
    
    Feature extracted at mixed-6 to have effective stride = 16
    
    Takes input_tensor as optional argument
    # do not change the arguments of the function
    
    """
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    channel_axis = 3
            
    
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid',trainable=trainable)
    x = conv2d_bn(x, 32, 3, 3, padding='valid',trainable=trainable)
    x = conv2d_bn(x, 64, 3, 3,trainable=trainable)
    x = MaxPooling2D((3, 3), strides=(2, 2),trainable=trainable)(x) #max pooling has no params anyway

    x = conv2d_bn(x, 80, 1, 1, padding='valid',trainable=trainable)
    x = conv2d_bn(x, 192, 3, 3, padding='valid',trainable=trainable)
    x = MaxPooling2D((3, 3), strides=(2, 2),trainable=trainable)(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1,trainable=trainable)

    branch5x5 = conv2d_bn(x, 48, 1, 1,trainable=trainable)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5,trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1,trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,trainable=trainable)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same',trainable=trainable)(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1,trainable=trainable)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1,trainable=trainable)

    branch5x5 = conv2d_bn(x, 48, 1, 1,trainable=trainable)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5,trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1,trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,trainable=trainable)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same',trainable=trainable)(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1,trainable=trainable)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1,trainable=trainable)

    branch5x5 = conv2d_bn(x, 48, 1, 1,trainable=trainable)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5,trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1,trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,trainable=trainable)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same',trainable=trainable)(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1,trainable=trainable)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid',trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1,trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,trainable=trainable)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid',trainable=trainable)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2),trainable=trainable)(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1,trainable=trainable)

    branch7x7 = conv2d_bn(x, 128, 1, 1,trainable=trainable)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7,trainable=trainable)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1,trainable=trainable)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1,trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1,trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7,trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1,trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7,trainable=trainable)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same',trainable=trainable)(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1,trainable=trainable)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1,trainable=trainable)

        branch7x7 = conv2d_bn(x, 160, 1, 1,trainable=trainable)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7,trainable=trainable)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1,trainable=trainable)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1,trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1,trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7,trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1,trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7,trainable=trainable)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same',trainable=trainable)(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1,trainable=trainable)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))
     
    # mixed 7: 17 x 17 x 768 (mixed6e in tensorflow) 
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')
        
    return(x)
    
    
def rpn(base_layers, num_anchors,trainable=True):
    
    """ RPN architecture Adopted from "Faster R-CNN: Towards Real-Time Object 
        Detection with Region Proposal Networks"
        
        512-depth feature-map
    
    """

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1',trainable=trainable)(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class',trainable=trainable)(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress',trainable=trainable)(x)

    return [x_class, x_regr, base_layers]


def conv2d_bn_td(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              trainable=True):
    """Utility function to apply conv + BN with Timedistributed layer.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = TimeDistributed(Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        trainable=trainable),name=conv_name,trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, scale=False,trainable=trainable), name=bn_name,trainable=trainable)(x)
    x = Activation('relu', name=name,trainable=trainable)(x) #this has no params anyway
    return x


def classifier(base_layers, input_rois, num_rois, nb_classes,trainable=True):
    """
    The final classifier
    NOTE:
    The Roipooling layer uses tensorflow's bilinear interpolation
    """
    channel_axis = 4 # additional TD layer
    pooling_regions = 17 # tensorflow implementation
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois,trainable=trainable)([base_layers, input_rois])

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn_td(out_roi_pool, 192, 1, 1,trainable=trainable)
    branch3x3 = conv2d_bn_td(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid',trainable=trainable)

    branch7x7x3 = conv2d_bn_td(out_roi_pool, 192, 1, 1,trainable=trainable)
    branch7x7x3 = conv2d_bn_td(branch7x7x3, 192, 1, 7,trainable=trainable)
    branch7x7x3 = conv2d_bn_td(branch7x7x3, 192, 7, 1,trainable=trainable)
    branch7x7x3 = conv2d_bn_td(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid',trainable=trainable)

    branch_pool = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2),trainable=trainable),trainable=trainable)(out_roi_pool)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9,10: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn_td(x, 320, 1, 1,trainable=trainable)

        branch3x3 = conv2d_bn_td(x, 384, 1, 1,trainable=trainable)
        branch3x3_1 = conv2d_bn_td(branch3x3, 384, 1, 3,trainable=trainable)
        branch3x3_2 = conv2d_bn_td(branch3x3, 384, 3, 1,trainable=trainable)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn_td(x, 448, 1, 1,trainable=trainable)
        branch3x3dbl = conv2d_bn_td(branch3x3dbl, 384, 3, 3,trainable=trainable)
        branch3x3dbl_1 = conv2d_bn_td(branch3x3dbl, 384, 1, 3,trainable=trainable)
        branch3x3dbl_2 = conv2d_bn_td(branch3x3dbl, 384, 3, 1,trainable=trainable)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = TimeDistributed(AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same',trainable=trainable),trainable=trainable)(x)
        branch_pool = conv2d_bn_td(branch_pool, 192, 1, 1,trainable=trainable)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    out = TimeDistributed(GlobalAveragePooling2D(trainable=trainable),name='global_avg_pooling',trainable=trainable)(x)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero',trainable=trainable), name='dense_class_{}'.format(nb_classes),trainable=trainable)(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero',trainable=trainable), name='dense_regress_{}'.format(nb_classes),trainable=trainable)(out)

    return [out_class, out_regr]
