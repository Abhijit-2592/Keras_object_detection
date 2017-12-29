from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import pickle

from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import data_generators
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras_frcnn.simple_parser import get_data
from keras_frcnn import losses as Losses
import config
import math

#%%

sys.setrecursionlimit(40000)

def Train_frcnn(train_path, # path to the text file containing the data
                network_arch, # the full faster rcnn network architecture object
                num_epochs, # num of epochs
                output_weight_path, # path to save the model_all.weights as hdf5
                preprocessing_function = None,
                config_filename="config.pickle", 
                input_weights_path=None,
                train_rpn = True,
                train_final_classifier = True,
                train_base_nn = True,
                losses_to_watch = ['rpn_cls','rpn_reg','final_cls','final_reg'],
                tb_log_dir="log", 
                num_rois=32, 
                horizontal_flips=False,
                vertical_flips=False, 
                rot_90=False,
                anchor_box_scales=[128, 256, 512],
                anchor_box_ratios=[[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]],
                im_size=600,
                rpn_stride=16, # depends on network architecture
                visualize_model=None,
                verify_trainable = True,
                optimizer_rpn = Adam(lr=1e-5),
                optimizer_classifier = Adam(lr=1e-5),
                validation_interval = 3,
                rpn_min_overlap = 0.3,
                rpn_max_overlap = 0.7,
                classifier_min_overlap = 0.1,
                classifier_max_overlap = 0.5,
                rpn_nms_threshold = 0.7, # original implementation
                seed=5000
                ):
    """
    Trains a Faster RCNN for object detection in keras
    
    NOTE: This trains 2 models namely model_rpn and model_classifer with the same shared base_nn (fixed feature extractor)
          
    Keyword Arguments
    train_path -- str: path to the text file or pascal_voc (no Default)
    network_arch --object: the full faster rcnn network .py file passed as an object (no default)
    num_epochs -- int: number of epochs to train (no Default)
    output_weight_path --str: path to save the frcnn weights (no Default)
    preprocessing_function --function: Optional preprocessing function (must be defined like given in keras docs) (Default None)
    config_filename --str: Path to save the config file. Used when testing (Default "config.pickle")
    input_weight_path --str: Path to hdf5 file containing weights for the base model (Default None)
    train_rpn --bool: whether to train the rpn layer (Default True)
    train_final_classifier --bool:Whether to train the final_classifier (Fast Rcnn layer) (Default True)
    train_base_nn --bool:Whether to train the base_nn/fixed_feature_extractor (Default True)
    losses_to_watch --list: A list of losses to watch (Default ['rpn_cls','rpn_reg','final_cls','final_reg']).
                            The losses in this list are added and then weights are saved wrt to that.
                            The list can contain any combination of the above 4 only.
    tb_log_dir --str: path to log dir for tensorboard logging (Default 'log')
    num_rois --int: The number of rois to use at once (Default = 32)
    horizontal_flips --bool: augment training data by horizontal flips (Default False)
    vertical_flips --bool: augment training data by vertical flips (Default False)
    rot_90 --bool: augment training data by 90 deg rotations (Default False)
    anchor_box_scales --list: The list of anchor box scales to use (Default [128,256,512])
    anchor_box ratios --list of list: The list of anchorbox aspect ratios to use (Default [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]])
    im_size --int: The size to resize the image (Default 600). This is the smallest side of Pascal VOC format
    rpn_stride --int: The stride for rpn (Default = 16)
    visualize_model --str: Path to save the model as .png file
    verify_trainable --bool: print layer wise names and prints if it is trainable or not (Default True)
    optimizer_rpn --keras.optimizer: The optimizer for rpn (Default Adam(lr=1e-5))
    optimizer_classifier --keras.optimizer: The optimizer for classifier (Default Adam(lr=1e-5))
    validation_interval --int: The frequency (in epochs) to do validation. supply 0 if no validation
    rpn_min_overlap --float: (0,1) The Min IOU in rpn layer (Default 0.3) (original implementation)
    rpn_max_overlap --float: (0,1) The max IOU in rpn layer (Default 0.7) (original implementation)
    classifier_min_overlap --float: (0,1) same as above but in final classifier (Default 0.1) (original implementation)
    classifier_max_overlap --float: (0,1) same as above (Default 0.5) (original implementation)
    rpn_nms_threshold --float :(0,1) The threshold above which to supress the bbox using Non max supression in rpn (Default 0.7)(from original implementation)
    seed --int: To seed the random shuffling of training data (Default = 5000)
    
    Performing alternating training:
    Use the train_rpn,train_final_classifier and train_base_nn arguments to accomplish
    alternating training
    
    OUTPUT:
    prints the training log. Does not return anything
    
    Save details:
    1.saves the weights of the full FRCNN model as .h5
    2.saves a tensorboard file
    3.saves the history of weights saved in ./saving_log.txt so that it can be known at which epoch the model is saved
    4.saves the model configuration as a .pickle file
    5.optionally saves the full FRCNN architecture as .png
    
    NOTE: 
    as of now the batch size = 1
    Prints loss = 0 for losses from model which is not being trained
    
    """
    check_list = ['rpn_cls','rpn_reg','final_cls','final_reg']
    for n in losses_to_watch:
        if n not in check_list:
            raise ValueError("unsupported loss the supported losses are: {}".format(check_list))

    if not train_rpn:
        if "rpn_cls" in losses_to_watch or "rpn_reg" in losses_to_watch:
            raise ValueError("Cannot watch rpn_cls and rpn_reg when train_rpn == False")
    if not train_final_classifier:
        if "final_cls" in losses_to_watch or "final_reg" in losses_to_watch:
            raise ValueError("cannot watch final_cls and final_reg when train_final_classifier == False")
    
    
    nn = network_arch
    random.seed(seed)
    np.random.seed(seed)
    

    # pass the settings from the function call, and persist them in the config object
    C = config.Config()
    C.rpn_max_overlap = rpn_max_overlap
    C.rpn_min_overlap = rpn_min_overlap
    C.classifier_min_overlap = classifier_min_overlap
    C.classifier_max_overlap = classifier_max_overlap
    C.anchor_box_scales = anchor_box_scales
    C.anchor_box_ratios = anchor_box_ratios
    C.im_size = im_size
    C.use_horizontal_flips = bool(horizontal_flips)
    C.use_vertical_flips = bool(vertical_flips)
    C.rot_90 = bool(rot_90)
    C.rpn_stride=rpn_stride
    C.rpn_nms_threshold = rpn_nms_threshold
    C.weights_all_path = output_weight_path
    C.num_rois = int(num_rois)

    # check if weight path was passed via command line
    if input_weights_path:
        C.initial_weights = input_weights_path

    all_imgs, classes_count, class_mapping = get_data(train_path)
    
    print("The class mapping is:")
    print(class_mapping)

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping


    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))

    with open(config_filename, 'wb') as config_f:
        pickle.dump(C,config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_filename))

    np.random.shuffle(all_imgs)


    train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'valid']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))


    
    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))
    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input,trainable = train_base_nn)
    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors,trainable = train_rpn)
    # define the classifier, built on base layers
    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, len(classes_count),trainable = train_final_classifier)
    # create models
    model_base = Model(img_input,shared_layers) # for computing the output shape
    model_rpn = Model(img_input, rpn[:2]) # used for training
    model_classifier = Model([img_input, roi_input], classifier) # used for training
    # this is a model that holds both the RPN and the classifier, used to load/save and freeze/unfreeze weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    # tensorboard
    tbCallBack = TensorBoard(log_dir=tb_log_dir, histogram_freq=1,write_graph=False, write_images=False)
    tbCallBack.set_model(model_all)
    
    #NOTE: both model_rpn and model_classifer contains the base_nn
    
    try:
        print('loading weights from {}'.format(C.initial_weights))
        model_all.load_weights(C.initial_weights, by_name=True)
    except:
        print('Could not load pretrained model weights')
    
     # number of trainable parameters
    trainable_count = int(np.sum([K.count_params(p) for p in set(model_all.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model_all.non_trainable_weights)]))
    
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    
    if verify_trainable:
        for layer in model_all.layers:
            print(layer.name,layer.trainable)
        
    model_rpn.compile(optimizer=optimizer_rpn, loss=[Losses.rpn_loss_cls(num_anchors), Losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[Losses.class_loss_cls, Losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mse')
    # save model_all as png for visualization
    if visualize_model != None:
        plot_model(model=model_all,to_file=visualize_model,show_shapes=True,show_layer_names=True)
        
            
    epoch_length = len(train_imgs)
    validation_epoch_length=len(val_imgs)
    num_epochs = int(num_epochs)
    iter_num = 0
    
    # train and valid data generator
    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, model_base, K.image_dim_ordering(), preprocessing_function ,mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, model_base,K.image_dim_ordering(), preprocessing_function ,mode='val')

    
    losses_val=np.zeros((validation_epoch_length,5))
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf
    val_best_loss = np.Inf
    val_best_loss_epoch = 0

    print('Starting training')
    
    def write_log(callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
    
    train_names = ['train_loss_rpn_cls', 'train_loss_rpn_reg','train_loss_class_cls','train_loss_class_reg','train_total_loss','train_acc']
    val_names = ['val_loss_rpn_cls', 'val_loss_rpn_reg','val_loss_class_cls','val_loss_class_reg','val_total_loss','val_acc']


    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)
                
                if train_rpn:
                    loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=C.rpn_nms_threshold,flag="train")
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                if train_final_classifier:
                    loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                
                # losses
                
                if train_rpn:
                    losses[iter_num, 0] = loss_rpn[1]
                    losses[iter_num, 1] = loss_rpn[2]
                else:
                    losses[iter_num, 0] = 0
                    losses[iter_num, 1] = 0
                    
                if train_final_classifier:
                    losses[iter_num, 2] = loss_class[1]
                    losses[iter_num, 3] = loss_class[2]
                    losses[iter_num, 4] = loss_class[3] # accuracy
                else:
                    losses[iter_num, 2] = 0
                    losses[iter_num, 3] = 0
                    losses[iter_num, 4] = 0
                    

                iter_num += 1

                progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                          ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    if train_rpn:
                        loss_rpn_cls = np.mean(losses[:, 0])
                        loss_rpn_regr = np.mean(losses[:, 1])
                    else:
                        loss_rpn_cls = 0
                        loss_rpn_regr = 0
                        
                    if train_final_classifier:
                        loss_class_cls = np.mean(losses[:, 2])
                        loss_class_regr = np.mean(losses[:, 3])
                        class_acc = np.mean(losses[:, 4])
                    else:
                        loss_class_cls = 0
                        loss_class_regr = 0
                        class_acc = 0

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))
                        
                    loss_dict_train = {"rpn_cls":loss_rpn_cls,"rpn_reg":loss_rpn_regr,"final_cls":loss_class_cls,"final_reg":loss_class_regr}
                    
                    curr_loss = 0
                    for l in losses_to_watch:
                        curr_loss += loss_dict_train[l]
                    
                    iter_num = 0
                    start_time = time.time()
                    write_log(tbCallBack, train_names, [loss_rpn_cls,loss_rpn_regr,loss_class_cls,loss_class_regr,curr_loss,class_acc], epoch_num)

                    if curr_loss < best_loss:
                        if C.verbose:
                            print('Total loss decreased from {} to {} in training, saving weights'.format(best_loss,curr_loss))
                            save_log_data = '\nTotal loss decreased from {} to {} in epoch {}/{} in training, saving weights'.format(best_loss,curr_loss,epoch_num + 1,num_epochs)
                            with open("./saving_log.txt","a") as f:
                                f.write(save_log_data)
                                
                        best_loss = curr_loss
                        model_all.save_weights(C.weights_all_path)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue
            
        if validation_interval > 0: 
            # validation
            if (epoch_num+1)%validation_interval==0 :
                progbar = generic_utils.Progbar(validation_epoch_length)
                print("Validation... \n")
                while True:
                    try:
                        X, Y, img_data = next(data_gen_val)
                        
                        if train_rpn:
                            val_loss_rpn = model_rpn.test_on_batch(X, Y)
            
                        P_rpn = model_rpn.predict_on_batch(X)
                        R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=C.rpn_nms_threshold,flag="train")
                        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
                        
                        neg_samples = np.where(Y1[0, :, -1] == 1)
                        pos_samples = np.where(Y1[0, :, -1] == 0)
            
                        if len(neg_samples) > 0:
                            neg_samples = neg_samples[0]
                        else:
                            neg_samples = []
            
                        if len(pos_samples) > 0:
                            pos_samples = pos_samples[0]
                        else:
                            pos_samples = []
                        
                        rpn_accuracy_rpn_monitor.append(len(pos_samples))
                        rpn_accuracy_for_epoch.append((len(pos_samples)))
            
                        if C.num_rois > 1:
                            if len(pos_samples) < C.num_rois//2:
                                selected_pos_samples = pos_samples.tolist()
                            else:
                                selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                            try:
                                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                            except:
                                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
            
                            sel_samples = selected_pos_samples + selected_neg_samples
                        else:
                            # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                            selected_pos_samples = pos_samples.tolist()
                            selected_neg_samples = neg_samples.tolist()
                            if np.random.randint(0, 2):
                                sel_samples = random.choice(neg_samples)
                            else:
                                sel_samples = random.choice(pos_samples)
                        if train_final_classifier:
                            val_loss_class = model_classifier.test_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                        
                        if train_rpn:
                            losses_val[iter_num, 0] = val_loss_rpn[1]
                            losses_val[iter_num, 1] = val_loss_rpn[2]
                        else:
                            losses_val[iter_num, 0] = 0
                            losses_val[iter_num, 1] = 0
                            
                        if train_final_classifier:
                            losses_val[iter_num, 2] = val_loss_class[1]
                            losses_val[iter_num, 3] = val_loss_class[2]
                            losses_val[iter_num, 4] = val_loss_class[3]
                        else:
                            losses_val[iter_num, 2] = 0
                            losses_val[iter_num, 3] = 0
                            losses_val[iter_num, 4] = 0
                            
            
                        iter_num += 1
            
                        progbar.update(iter_num, [('rpn_cls', np.mean(losses_val[:iter_num, 0])), ('rpn_regr', np.mean(losses_val[:iter_num, 1])),
                                                  ('detector_cls', np.mean(losses_val[:iter_num, 2])), ('detector_regr', np.mean(losses_val[:iter_num, 3]))])
            
                        if iter_num == validation_epoch_length:
                            if train_rpn:
                                val_loss_rpn_cls = np.mean(losses_val[:, 0])
                                val_loss_rpn_regr = np.mean(losses_val[:, 1])
                            else:
                                val_loss_rpn_cls = 0
                                val_loss_rpn_regr = 0
                            if train_final_classifier:
                                val_loss_class_cls = np.mean(losses_val[:, 2])
                                val_loss_class_regr = np.mean(losses_val[:, 3])
                                val_class_acc = np.mean(losses_val[:, 4])
                            else:
                                val_loss_class_cls = 0
                                val_loss_class_regr = 0
                                val_class_acc = 0
                                
            
                            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                            rpn_accuracy_for_epoch = []
                            
                            loss_dict_valid = {"rpn_cls":val_loss_rpn_cls,"rpn_reg":val_loss_rpn_regr,"final_cls":val_loss_class_cls,"final_reg":val_loss_class_regr}
                    
                            val_curr_loss = 0
                            for l in losses_to_watch:
                                val_curr_loss += loss_dict_valid[l]
                                 
                            write_log(tbCallBack, val_names, [val_loss_rpn_cls,val_loss_rpn_regr,val_loss_class_cls,val_loss_class_regr,val_curr_loss,val_class_acc], epoch_num)
            
                            if C.verbose:
                                print('[INFO VALIDATION]')
                                print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                                print('Classifier accuracy for bounding boxes from RPN: {}'.format(val_class_acc))
                                print('Loss RPN classifier: {}'.format(val_loss_rpn_cls))
                                print('Loss RPN regression: {}'.format(val_loss_rpn_regr))
                                print('Loss Detector classifier: {}'.format(val_loss_class_cls))
                                print('Loss Detector regression: {}'.format(val_loss_class_regr))
                                print("current loss: %.2f, best loss: %.2f at epoch: %d"%(val_curr_loss,val_best_loss,val_best_loss_epoch))
                                print('Elapsed time: {}'.format(time.time() - start_time))               
            
                            if val_curr_loss < val_best_loss:
                                if C.verbose:
                                    print('Total loss decreased from {} to {}, saving weights'.format(val_best_loss,val_curr_loss))
                                    save_log_data = '\nTotal loss decreased from {} to {} in epoch {}/{} in validation, saving weights'.format(val_best_loss,val_curr_loss,epoch_num + 1 ,num_epochs)
                                    with open("./saving_log.txt","a") as f:
                                        f.write(save_log_data)
                                val_best_loss = val_curr_loss
                                val_best_loss_epoch=epoch_num
                                model_all.save_weights(C.weights_all_path)
                            start_time = time.time()
                            iter_num = 0
                            break
                    except:
                        pass
        
        

    print('Training complete, exiting.')
    
    
    
    
