# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:23:10 2017

@author: Abhijit

NOTE:
In this file you have to define atleast 3 functions namely : 
    1) nn_base()
    2) rpn()
    3) classifier()
    
Make sure you don't change the function names and the keyword arguments

STEPS TO DEFINE NEW FEATURE EXTRACTOR ARCHITECTURE 
1) Create a new foo.py file
2) Define the 3 functions stated above (Those are mandatory)
    -- nn_base() is the first stage feature extractor
    -- rpn() is the RPN ;ayer
    -- classifier() is the final classification layer
3) Make sure you wrap the layers in the final classifier layer using TimeDistributed() layer 
   so that it can process multiple batches of ROIs simultaneously
4) Refer nn_arch_inceptionv3.py for additional example  
    
Pass this whole foo.py as an object to Train_frcnn method
"""
############### define VGG-16 model for faster_rcnn########################
# future imports
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras.layers import Flatten, Dense, Dropout, TimeDistributed
from keras.layers import Input, Conv2D, MaxPooling2D
from keras import backend as K


def nn_base(input_tensor=None,trainable=True):
    """ The architecture of VGG-16 (fixed feature extractor)
    
    Takes input_tensor as optional argument
    # do not change the arguments of the function
    
    NOTE: Make sure to give names for all the layers. These names will be
          used to freeze the corresponding layers if doing 4-step alternating training
            
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

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',trainable=trainable)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',trainable=trainable)(x) # max pooling has no trainbale layers

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',trainable=trainable)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',trainable=trainable)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',trainable=trainable)(x)
    
    return(x)
    
def rpn(base_layers, num_anchors,trainable=True):
    
    """ RPN architecture to match original implementaion for VGG-16
    
    """

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1',trainable=trainable)(base_layers)# need to be changed to 512

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class',trainable=trainable)(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress',trainable=trainable)(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes,trainable=True):
    """
    The final classifier to match original implementation for VGG-16
    The only difference being the Roipooling layer uses tensorflow's bilinear interpolation
    """
    
    pooling_regions = 7
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois,trainable=trainable)([base_layers, input_rois])

    out = TimeDistributed(Flatten(),name="flatten",trainable=trainable)(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu',trainable=trainable),name="fc1",trainable=trainable)(out)
    out = TimeDistributed(Dropout(0.5),name="drop_out1",trainable=trainable)(out) # add dropout to match original implememtation
    out = TimeDistributed(Dense(4096, activation='relu',trainable=trainable),name="fc2",trainable=trainable)(out)
    out = TimeDistributed(Dropout(0.5),name="drop_out2",trainable=trainable)(out) # add dropout to match original implementation

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero',trainable=trainable), name='dense_class_{}'.format(nb_classes),trainable=trainable)(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero',trainable=trainable), name='dense_regress_{}'.format(nb_classes),trainable=trainable)(out)

    return [out_class, out_regr]