from __future__ import division
import cv2
import numpy as np
import sys
import pickle
import time
from keras import backend as K
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score
from keras_frcnn.simple_parser import get_data
from keras.layers import Input
from keras.models import Model
sys.setrecursionlimit(40000)


def Measure_map(test_path, 
                network_arch,
                config_filename, 
                preprocessing_function = None,
                mAP_threshold = 0.5
                ):
    """Function to measure Mean Average prediction metric for object detection

    Keyword Arguments
    test_path --str: Path to the .txt file of testing data (No default)
    network_arc --object: the full faster rcnn network .py file passed as an object (no default)
    config_filename --str: Path to config file (No default)
    preprocessing_function --function: optional image preprocessing function (Default None)
    mAP threshold --float: (0,1) The min threshold to consider as a correct prediction (default 0.5)

    Output:
    prints the Map on the test dataset and returns a list of all Maps
    """
    nn = network_arch

    def get_map(pred, gt, f,threshold):
        T = {}
        P = {}
        fx, fy = f

        for bbox in gt:
            bbox['bbox_matched'] = False

        pred_probs = np.array([s['prob'] for s in pred])
        box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

        for box_idx in box_idx_sorted_by_prob:
            pred_box = pred[box_idx]
            pred_class = pred_box['class']
            pred_x1 = pred_box['x1']
            pred_x2 = pred_box['x2']
            pred_y1 = pred_box['y1']
            pred_y2 = pred_box['y2']
            pred_prob = pred_box['prob']
            if pred_class not in P:
                P[pred_class] = []
                T[pred_class] = []
            P[pred_class].append(pred_prob)
            found_match = False

            for gt_box in gt:
                gt_class = gt_box['class']
                gt_x1 = gt_box['x1']/fx
                gt_x2 = gt_box['x2']/fx
                gt_y1 = gt_box['y1']/fy
                gt_y2 = gt_box['y2']/fy
                gt_seen = gt_box['bbox_matched']
                if gt_class != pred_class:
                    continue
                if gt_seen:
                    continue
                iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                if iou >= threshold: #0.5 default
                    found_match = True
                    gt_box['bbox_matched'] = True
                    break
                else:
                    continue

            T[pred_class].append(int(found_match))

        for gt_box in gt:
            if not gt_box['bbox_matched']:
                if gt_box['class'] not in P:
                    P[gt_box['class']] = []
                    T[gt_box['class']] = []

                T[gt_box['class']].append(1)
                P[gt_box['class']].append(0)

        #import pdb
        #pdb.set_trace()
        return T, P


    with open(config_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    def format_img(img, C,preprocessing_function):
        img_min_side = float(C.im_size)
        (height,width,_) = img.shape

        if width <= height:
            f = img_min_side/width
            new_height = int(f * height)
            new_width = int(img_min_side)
        else:
            f = img_min_side/height
            new_width = int(f * width)
            new_height = int(img_min_side)
        fx = width/float(new_width)
        fy = height/float(new_height)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        img = img[:, :, (2, 1, 0)] #bgr to RGB
        if preprocessing_function:
            img = preprocessing_function(img)
        # img = np.transpose(img, (2, 0, 1)) theano format
        img = np.expand_dims(img, axis=0)
        return img, fx, fy

    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    
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
    # This is done to increase the testing speed

    print('Loading weights from {}'.format(C.weights_all_path))
    model_rpn.load_weights(C.weights_all_path, by_name=True)
    model_classifier.load_weights(C.weights_all_path, by_name=True)


    test_imgs, _, _ = get_data(test_path)
    
    T = {}
    P = {}
    ALL_MAP_LIST = []
    for idx, img_data in enumerate(test_imgs):
        print('{}/{}'.format(idx,len(test_imgs)))
        st = time.time()
        filepath = img_data['filepath']

        img = cv2.imread(filepath)

        X, fx, fy = format_img(img, C,preprocessing_function)

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=C.rpn_nms_threshold,flag="test")

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=C.test_roi_nms_threshold,max_boxes=C.TEST_RPN_POST_NMS_TOP_N)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                all_dets.append(det)


        print('Elapsed time = {}'.format(time.time() - st))
        t, p = get_map(all_dets, img_data['bboxes'], (fx, fy),mAP_threshold)
        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])
        all_aps = []
        for key in T.keys():
            ap = average_precision_score(T[key], P[key])
            print('{} AP: {}'.format(key, ap))
            all_aps.append(ap)
        print('mAP = {}'.format(np.mean(np.array(all_aps))))
        ALL_MAP_LIST.append(np.mean(np.array(all_aps)))
    return(ALL_MAP_LIST)
        #print(T)
        #print(P)
    