from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from keras import backend as K
from keras_frcnn import roi_helpers
import pandas as pd
from keras.layers import Input
from keras.models import Model

sys.setrecursionlimit(40000)

def Test_frcnn(test_images_list,  
               network_arch,
               config_filename,
               preprocessing_function = None,
               num_rois = None,
               final_classification_threshold = 0.8):
    
    """
    Test the object detection network
    
    test_images_list --list: list containing path to test_images (No default)
    network_arc --object: the full faster rcnn network .py file passed as an object (no default)
    config_filename --str: Full path to the config_file.pickle, generated while training (No default)
    preprocessing_function --function: optional image preprocessing function (Default None)
    num_rois --int: (optional)The number of ROIs to process at once in the final classifier (Default None)
                    if not given. The number of ROIs given while training is chosen
    final_classification_threshold --float: (0,1) min threshold for accepting as a detection in final classifier (Default 0.8)                       
    
    OUTPUT:
    returns the images with bboxes over layed using opencv, and a dataframe with data
    """
    nn = network_arch


    assert "list" in str(type(test_images_list)),"test_images_list must be a list of paths to the test images"

    with open(config_filename, 'rb') as f_in:
        C = pickle.load(f_in)
    if num_rois:
        C.num_rois = int(num_rois)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    def format_img_size(img, C): # utility function 1
        """ formats the image size based on config """
        img_min_side = float(C.im_size)
        (height,width,_) = img.shape

        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio	

    def preprocess_img(img, preprocessing_function): #utility function 2
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)] #bgr to rgb
        if preprocessing_function:
            img = preprocessing_function(img)
        #img = np.transpose(img, (2, 0, 1)) # convert to theano
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(img, C, preprocessing_function): # utility function 3
        """ formats an image for model prediction based on config """
        img, ratio = format_img_size(img, C)
        img = preprocess_img(img, preprocessing_function)
        return img, ratio

    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(ratio, x1, y1, x2, y2): #utility function 4

        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)

    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    # load the models
    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))
    shared_layers = nn.nn_base(img_input)
    
    num_features = shared_layers.get_shape().as_list()[3] #512 for vgg-16
    feature_map_input = Input(shape=(None, None, num_features))
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, len(class_mapping))
    # create a keras model
    model_rpn = Model(img_input, rpn)
    model_classifier = Model([feature_map_input, roi_input], classifier)
    
    #Note: The model_classifier in training and testing are different.
    # In training model_classifier and model_rpn both have the base_nn.
    # while testing only model_rpn has the base_nn it returns the FM of base_nn
    # Thus the model_classifier has the FM and ROI as input
    # This id done to increase the testing speed
    
    print('Loading weights from {}'.format(C.weights_all_path))
    model_rpn.load_weights(C.weights_all_path, by_name=True)
    model_classifier.load_weights(C.weights_all_path, by_name=True)

    
    list_of_all_images=[]
    df_list = []
    
    for idx, filepath in enumerate(sorted(test_images_list)):
        print(os.path.basename(filepath))

        img = cv2.imread(filepath)

        X, ratio = format_img(img, C, preprocessing_function)

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)


        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=C.rpn_nms_threshold,flag="test")

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < final_classification_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        probs_list = [] # new list for every image
        coor_list = [] # new list for every image
        classes_list = []# new list for every image
        img_name_list = []# new list for ever image
        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=C.test_roi_nms_threshold,max_boxes=C.TEST_RPN_POST_NMS_TOP_N) #0.3 default threshold from original implementation
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                coor_list.append([real_x1,real_y1,real_x2,real_y2]) # get the coordinates
                classes_list.append(key)
                probs_list.append(100*new_probs[jk])
                img_name_list.append(filepath)

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            
        df = pd.DataFrame({"Image_name":img_name_list,
                           "classes":classes_list,
                           "pred_prob":probs_list, 
                           "x1_y1_x2_y2":coor_list})
            
    
        
        list_of_all_images.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        df_list.append(df)
            
    final_df = pd.concat(df_list,ignore_index=True)
        
    return(list_of_all_images,final_df)


    