from libraries import *
from recorder import Recorder
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def train_model(train_imgs, num_epochs, record_filepath):
    recorder = Recorder(record_filepath)
    losses = np.zeros((len(train_imgs), 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    for epoch_num in range(num_epochs):
        start_time = time.time()
        progbar = generic_utils.Progbar(len(train_imgs))
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
        random.shuffle(train_imgs)
        data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length,
                                       mode='train')  # TODO: tester mode:'augmentation'

        for iter_num in range(len(train_imgs)):
            try:

                # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                X, Y, img_data, _, _ = next(data_gen_train)

                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                loss_rpn = model_rpn.train_on_batch(X, Y)
                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                X2, Y1, Y2, sel_samples = rpn_to_class(X, img_data, rpn_accuracy_rpn_monitor, rpn_accuracy_for_epoch)

                if X2 is None:
                    continue

                # training_data: [X, X2[:, sel_samples, :]]
                # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1,
                               [('rpn_cls', np.mean(losses[:iter_num + 1, 0])),
                                ('rpn_regr', np.mean(losses[:iter_num + 1, 1])),
                                ('final_cls', np.mean(losses[:iter_num + 1, 2])),
                                ('final_regr', np.mean(losses[:iter_num + 1, 3])),
                                ('loss', np.mean(losses[:iter_num + 1, 0])
                                 + np.mean(losses[:iter_num + 1, 1])
                                 + np.mean(losses[:iter_num + 1, 2])
                                 + np.mean(losses[:iter_num + 1, 3]))])

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

        loss_rpn_cls = np.mean(losses[:, 0])
        loss_rpn_regr = np.mean(losses[:, 1])
        loss_class_cls = np.mean(losses[:, 2])
        loss_class_regr = np.mean(losses[:, 3])
        class_acc = np.mean(losses[:, 4])

        if len(rpn_accuracy_for_epoch) == 0:
            mean_overlapping_bboxes = 0
        else:
            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)

        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
        elapsed_time = (time.time() - start_time)

        if C.verbose:
            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                mean_overlapping_bboxes))
            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
            print('Loss RPN regression: {}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {}'.format(loss_class_cls))
            print('Loss Detector regression: {}'.format(loss_class_regr))
            print('Total loss: {}'.format(curr_loss))
            print('Elapsed time: {}'.format(elapsed_time))

        model_all.save_weights(C.temp_model_path)
        recorder.add_new_entry(class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr,
                               curr_loss, elapsed_time)

    # recorder.show_graphs()
    recorder.save_graphs()
    model_all.save_weights(C.model_path)
    os.remove(C.temp_model_path)


def val_model(train_imgs, val_imgs, param, paramNames, record_path, validation_code):
    global last_epoch

    for i in range(len(paramNames)):
        if paramNames[i] == "box_filter_shape":
            C.box_filter_shape = param[i]
            C.box_filter = param[i] is not None
        elif paramNames[i] == "brightness_jitter":
            C.use_brightness_jitter = param[i]
        elif paramNames[i] == "gamma_correction":
            C.gamma_correction = param[i]
        elif paramNames[i] == "histogram_equalization":
            C.histogram_equalization = param[i]
        else:
            pass

    recorder = Recorder(os.path.join(record_path, validation_code), has_validation=True)
    losses = np.zeros((len(train_imgs), 5))
    losses_val = np.zeros((len(val_imgs), 5))
    best_loss_val = float('inf')
    curr_loss_val = float('inf')
    best_epoch = -1

    for epoch_num in range(num_epochs):
        if (not last_epoch == 0) and (not epoch_num == last_epoch):
            continue
        else:
            if not last_epoch == 0:
                last_row = validation_record_df.tail(1)
                best_loss_val = list(last_row['best_loss'])[-1]
            last_epoch = 0
        start_time = time.time()
        progbar = generic_utils.Progbar(len(train_imgs))
        progbar_val = generic_utils.Progbar(len(val_imgs))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
        # TODO: tester mode:'augmentation'
        data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length, mode='train')

        for iter_num in range(len(train_imgs)):
            try:

                # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                X, Y, img_data, _, _ = next(data_gen_train)

                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                loss_rpn = model_rpn.train_on_batch(X, Y)
                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                X2, Y1, Y2, sel_samples = rpn_to_class(X, img_data, rpn_accuracy_rpn_monitor, rpn_accuracy_for_epoch)

                if X2 is None:
                    continue

                # training_data: [X, X2[:, sel_samples, :]]
                # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1,
                               [('rpn_cls', np.mean(losses[:iter_num + 1, 0])),
                                ('rpn_regr', np.mean(losses[:iter_num + 1, 1])),
                                ('final_cls', np.mean(losses[:iter_num + 1, 2])),
                                ('final_regr', np.mean(losses[:iter_num + 1, 3])),
                                ('loss', np.mean(losses[:iter_num + 1, 0])
                                 + np.mean(losses[:iter_num + 1, 1])
                                 + np.mean(losses[:iter_num + 1, 2])
                                 + np.mean(losses[:iter_num + 1, 3]))])

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

        loss_rpn_cls = np.mean(losses[:, 0])
        loss_rpn_regr = np.mean(losses[:, 1])
        loss_class_cls = np.mean(losses[:, 2])
        loss_class_regr = np.mean(losses[:, 3])
        class_acc = np.mean(losses[:, 4])

        if len(rpn_accuracy_for_epoch) == 0:
            mean_overlapping_bboxes = 0
        else:
            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)

        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
        elapsed_time = (time.time() - start_time)

        if C.verbose:
            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                mean_overlapping_bboxes))
            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            print('Loss RPN classifier: {}'.format(loss_rpn_cls))
            print('Loss RPN regression: {}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {}'.format(loss_class_cls))
            print('Loss Detector regression: {}'.format(loss_class_regr))
            print('Total loss: {}'.format(curr_loss))
            print('Elapsed time: {}'.format(elapsed_time))

        print('Start of the validation phase')
        data_gen_val = get_anchor_gt(val_imgs, C, get_img_output_length, mode='train')

        for iter_num_val in range(len(val_imgs)):
            try:
                # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                X_val, Y_val, img_data_val, _, _ = next(data_gen_val)
                loss_rpn_val = model_rpn.test_on_batch(X_val, Y_val)
                losses_val[iter_num_val, 0] = loss_rpn_val[1]
                losses_val[iter_num_val, 1] = loss_rpn_val[2]

                X2_val, Y1_val, Y2_val, sel_samples_val = rpn_to_class(X_val, img_data_val, [], [])

                if X2_val is None:
                    continue

                # training_data: [X, X2[:, sel_samples, :]]
                # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                loss_class_val = model_classifier.test_on_batch([X_val, X2_val[:, sel_samples_val, :]],
                                                                [Y1_val[:, sel_samples_val, :],
                                                                 Y2_val[:, sel_samples_val, :]])

                losses_val[iter_num_val, 2] = loss_class_val[1]
                losses_val[iter_num_val, 3] = loss_class_val[2]
                losses_val[iter_num_val, 4] = loss_class_val[3]

                progbar_val.update(iter_num_val + 1,
                                   [('rpn_cls_val', np.mean(losses_val[:iter_num_val + 1, 0])),
                                    ('rpn_regr_val', np.mean(losses_val[:iter_num_val + 1, 1])),
                                    ('final_cls_val', np.mean(losses_val[:iter_num_val + 1, 2])),
                                    ('final_regr_val', np.mean(losses_val[:iter_num_val + 1, 3])),
                                    ('loss_val', np.mean(losses_val[:iter_num_val + 1, 0])
                                     + np.mean(losses_val[:iter_num_val + 1, 1])
                                     + np.mean(losses_val[:iter_num_val + 1, 2])
                                     + np.mean(losses_val[:iter_num_val + 1, 3]))])

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

        print('End of the validation phase')

        loss_rpn_cls_val = np.mean(losses_val[:, 0])
        loss_rpn_regr_val = np.mean(losses_val[:, 1])
        loss_class_cls_val = np.mean(losses_val[:, 2])
        loss_class_regr_val = np.mean(losses_val[:, 3])
        class_acc_val = np.mean(losses_val[:, 4])
        curr_loss_val = loss_rpn_cls_val + loss_rpn_regr_val + loss_class_cls_val + loss_class_regr_val

        if C.verbose:
            print('Validation classifier accuracy for bounding boxes from RPN: {}'.format(class_acc_val))
            print('Validation loss RPN classifier: {}'.format(loss_rpn_cls_val))
            print('Validation loss RPN regression: {}'.format(loss_rpn_regr_val))
            print('Validation loss Detector classifier: {}'.format(loss_class_cls_val))
            print('Validation loss Detector regression: {}'.format(loss_class_regr_val))
            print('Total validation loss: {}'.format(curr_loss_val))

        if curr_loss_val <= best_loss_val:
            if C.verbose:
                print('Total validation loss decreased from {} to {}'.format(best_loss_val,
                                                                             curr_loss_val))
            best_loss_val = curr_loss_val
            best_epoch = epoch_num + 1
            model_all.save_weights(os.path.join(record_path, validation_code + ".hdf5"))

        new_row = {'validation_code': validation_code,
                       'curr_loss': curr_loss_val,
                       'best_loss': best_loss_val,
                       'best_epoch': best_epoch}

        record_df = validation_record_df.append(new_row, ignore_index=True)
        record_df.to_csv(validation_record_path, index=0)

        recorder.add_new_entry_with_validation(class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls,
                                               loss_class_regr, curr_loss, elapsed_time, class_acc_val,
                                               loss_rpn_cls_val, loss_rpn_regr_val, loss_class_cls_val,
                                               loss_class_regr_val, curr_loss_val, best_loss_val)

    # recorder.show_graphs()
    recorder.save_graphs()
    return curr_loss_val, best_loss_val, best_epoch


def rpn_to_class(X, img_data, rpn_accuracy_rpn_monitor, rpn_accuracy_for_epoch):
    # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
    P_rpn = model_rpn.predict_on_batch(X)

    # R: bboxes (shape=(300,4))
    # Convert rpn layer to roi bboxes
    R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_data_format(), use_regr=True, overlap_thresh=0.7,
                   max_boxes=300)

    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
    # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    # Y1: one hot code for bboxes from above => x_roi (X)
    # Y2: corresponding labels and corresponding gt bboxes
    X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)

    # If X2 is None means there are no matching bboxes
    if X2 is None:
        rpn_accuracy_rpn_monitor.append(0)
        rpn_accuracy_for_epoch.append(0)
        return None, None, None, None

    # Find out the positive anchors and negative anchors
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

    # TODO: do something with these measurements
    rpn_accuracy_rpn_monitor.append(len(pos_samples))
    rpn_accuracy_for_epoch.append((len(pos_samples)))

    if C.num_rois > 1:
        # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
        if len(pos_samples) < C.num_rois // 2:
            selected_pos_samples = pos_samples.tolist()
        else:
            selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()

        # Randomly choose (num_rois - num_pos) neg samples
        try:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                    replace=False).tolist()
        except:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                    replace=True).tolist()

        # Save all the pos and neg samples in sel_samples
        sel_samples = selected_pos_samples + selected_neg_samples
    else:
        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
        selected_pos_samples = pos_samples.tolist()
        selected_neg_samples = neg_samples.tolist()
        if np.random.randint(0, 2):
            sel_samples = random.choice(neg_samples)
        else:
            sel_samples = random.choice(pos_samples)

    return X2, Y1, Y2, sel_samples


"""
    Define the base network (VGG)
"""


def initialize_model():
    # TODO: maybe we need to simply reload the pretrained vgg weights and call this method only once

    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    shared_layers = nn_base(img_input, trainable=True)
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)  # 9

    # Define the RPN, built on the base layers
    rpn = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

    base_model_rpn = Model(img_input, rpn[:2])
    base_model_classifier = Model([img_input, roi_input], classifier)

    # This is a model that holds both the RPN and the classifier, used to load/save weights for the models
    base_model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    return base_model_all, base_model_rpn, base_model_classifier


def load_weights(weights_path):
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)  # 9
    try:
        print('Loading weights from {}'.format(weights_path))
        model_rpn.load_weights(weights_path, by_name=True)
        model_classifier.load_weights(weights_path, by_name=True)
    except Exception as e:
        print('Exception: {}'.format(e))

    model_rpn.compile(optimizer=Adam(lr=1e-5), loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=Adam(lr=1e-5),
                                  loss=[class_loss_cls, class_loss_regr(len(classes_count) - 1)],
                                  metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')


def split_imgs(imgs, val_split, test_split):
    train_split = 1 - val_split - test_split
    recalc_val_split = val_split / (val_split + test_split)
    train_list, temp = imgs[:int(len(imgs) * train_split)], imgs[int((len(imgs) * (train_split))):]
    val_list, test_list = temp[:int(len(temp) * recalc_val_split)], temp[int(len(temp) * recalc_val_split):]
    print("Train set size: {}\nVal set size: {}\nTest set size: {}\nTotal dataset size: {}".format(len(train_list), len(val_list), len(test_list), len(imgs)))
    return train_list, val_list, test_list


def get_validation_code(names, values):
    values_str = []
    for value in values:
        values_str.append(str(value))

    return "Validation - " + ", ".join(names) + " - " + ", ".join(values_str)


if __name__ == "__main__":

    TESTING_SPLIT = 0.2
    VALIDATION_SPLIT = 0.2
    TRAINING_SPLIT = 1 - TESTING_SPLIT - VALIDATION_SPLIT

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Use Faster R-CNN to detect insects.')
    parser.add_argument('--model_name', required=False,
                        metavar="name_of_your_model", default='model',
                        help='Name of the model being tested')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/insects/dataset/", default='./data',
                        help='Directory of the Insects dataset')
    parser.add_argument('--annotations', required=False, default='./data/data_annotations.txt',
                        metavar="/path/to/insects/dataset/annotations/file.txt",
                        help='Annotation file for the provided dataset')
    parser.add_argument('--validation', required=False,
                        metavar="True/False", default=False,
                        help='True to do a validation')
    parser.add_argument('--validation_code', required=False,
                        metavar="Validation code from last validation step", default=None,
                        help='Validation code from last validation step to provide if starting from where the '
                             'validation was left off')
    parser.add_argument('--start_from_epoch', required=False,
                        metavar="Epoch number (starting from 0) of where to restart the validation", default=0,
                        help='Epoch to start from if starting from where the '
                             'validation was left off')
    parser.add_argument('--num_epochs', required=False,
                        metavar="Integer", default=10,
                        help='Number of epochs for the training if --validation has been set to False\n'
                             'Maximum number of epochs for the validation if --validation has been set to True')
    parser.add_argument('--weights', required=False, default='./model/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                        metavar="/path/to/base/weights.h5",
                        help="Path to base weights .h5 file")
    parser.add_argument('--use_gpu', required=False, default="True",
                        metavar="True/False",
                        help="True if you want to run the training on a gpu, False otherwise")
    args = parser.parse_args()

    use_gpu = eval(args.use_gpu)
    if use_gpu is True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        session = tf.compat.v1.InteractiveSession(config=config)

    num_epochs = int(args.num_epochs)

    validation_record_path = "./logs/{}.csv".format(args.model_name)
    imgs_record_path = "./logs/{} - imgs.csv".format(args.model_name)

    last_epoch = int(args.start_from_epoch)
    last_validation_code = args.validation_code 

    if last_validation_code is not None:
        start_from_last_step = True
        imgs_record_df = pd.read_csv(imgs_record_path)       
        validation_record_df = pd.read_csv(validation_record_path)
    else:
        start_from_last_step = False
        if not last_epoch == 0:
            imgs_record_df = pd.read_csv(imgs_record_path)
            validation_record_df = pd.read_csv(validation_record_path)
        else:
            imgs_record_df = pd.DataFrame(columns=['train', 'val', 'test'])
            validation_record_df = pd.DataFrame(columns=['validation_code', 'curr_loss', 'best_loss', 'best_epoch'])

    train_path = args.annotations  # Training data (annotation file)
    data_path = args.dataset

    num_rois = 4  # Number of RoIs to process at once.

    # Augmentation flag
    horizontal_flips = True  # Augment with horizontal flips in training.
    vertical_flips = True  # Augment with vertical flips in training.
    rot_90 = True  # Augment with 90 degree rotations in training.
    brightness_jitter = True  # Augment with brightness jitter in training.

    # Record data (used to save the losses, classification accuracy and mean average precision)
    record_path = "./logs/{}".format(args.model_name)
    if not os.path.exists(record_path):
        os.mkdir(record_path)
    output_weight_path = "./model/{}.hdf5".format(args.model_name)
    output_temp_weight_path = "./model/temp_{}.hdf5".format(args.model_name)
    base_weight_path = args.weights
    config_output_filename = "./config/{}.pickle".format(args.model_name)

    C = Config()

    C.use_horizontal_flips = horizontal_flips
    C.use_vertical_flips = vertical_flips
    C.rot_90 = rot_90
    C.use_brightness_jitter = brightness_jitter

    C.record_path = record_path
    C.model_path = output_weight_path
    C.temp_model_path = output_temp_weight_path
    C.num_rois = num_rois

    C.base_net_weights = base_weight_path

    C.im_size = 300

    # C.im_size = 100
    # C.anchor_box_scales = [16, 32, 64] #Values should be < im_size
    
    st = time.time()
    all_imgs, classes_count, class_mapping = get_data(train_path, data_path)
    print()
    print('Spend %0.2f mins to load the data' % ((time.time() - st) / 60))

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    # e.g.
    #    classes_count: {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
    #    class_mapping: {'Person': 0, 'Car': 1, 'Mobile phone': 2, 'bg': 3}
    C.class_mapping = class_mapping

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    print(class_mapping)

    # Save the configuration
    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            config_output_filename))

    random.seed(1)

    print('Num train samples (images) {}'.format(len(all_imgs)))

    import itertools as it

    param = {
        'histogram_equalization': [True, False],
        'box_filter_shape': [(2,2), (3,3), None],
        'brightness_jitter': [True, False],
        'gamma_correction': [True, False]
    }

    paramNames = list(param.keys())
    combinations = it.product(*(param[Name] for Name in paramNames))

    model_all, model_rpn, model_classifier = initialize_model()
    
    if start_from_last_step or not last_epoch == 0:
        last_row = imgs_record_df.tail(1)
        train_imgs = ast.literal_eval(last_row['train'].tolist()[0])
        val_imgs = ast.literal_eval(last_row['val'].tolist()[0])
        test_imgs = ast.literal_eval(last_row['test'].tolist()[0])
    else:
        random.shuffle(all_imgs)
        train_imgs, val_imgs, test_imgs = split_imgs(all_imgs, VALIDATION_SPLIT, TESTING_SPLIT)
        new_row = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
        imgs_record_df = imgs_record_df.append(new_row, ignore_index=True)
        imgs_record_df.to_csv(imgs_record_path, index=0)

    best_values = {}
    if args.validation:
        for params in combinations:

            best_loss = float('inf')

            validation_code = get_validation_code(paramNames, list(params))

            if start_from_last_step:
                if not last_validation_code == validation_code:
                    continue
                else:
                    start_from_last_step = False
                    continue

            if not last_epoch == 0:
                load_weights(os.path.join(record_path, validation_code + ".hdf5"))
            else:
                load_weights(C.base_net_weights)

            print("Best loss: {}".format(best_loss))
            print("=== Validation step code: {}".format(validation_code))
            curr_loss_val, best_loss_val, best_epoch = val_model(train_imgs, val_imgs,
                                                                 params, paramNames,
                                                                 record_path, validation_code)

            if best_loss_val < best_loss:
                best_loss = best_loss_val
                for i in range(len(params)):
                    best_values[list(paramNames)[i]] = params[i]

        print("=== Best values:")
        for key in best_values.keys():
            print("    - {}: {}".format(key, best_values[key]))

    else:
        num_epochs = args.num_epochs
        train_model(all_imgs, math.ceil(num_epochs), os.path.join(C.record_path, "Training"))

    print('Training complete, exiting.')
