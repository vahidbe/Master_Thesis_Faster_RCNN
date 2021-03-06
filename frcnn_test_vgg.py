from libraries import *


# def plot_some_graphs(C):
#     # Load the records
#     record_df = pd.read_csv(C.record_path)
#
#     r_epochs = len(record_df)
#
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
#     plt.title('mean_overlapping_bboxes')
#
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
#     plt.title('class_acc')
#
#     plt.show()
#
#     plt.figure(figsize=(15, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
#     plt.title('loss_rpn_cls')
#
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
#     plt.title('loss_rpn_regr')
#     plt.show()
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
#     plt.title('loss_class_cls')
#
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
#     plt.title('loss_class_regr')
#     plt.show()
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(np.arange(0, r_epochs), record_df['curr_loss_classifier'], 'r')
#     plt.title('total_loss_classifer')
#
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
#     plt.title('elapsed_time')
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(np.arange(0, r_epochs), record_df['curr_loss_rpn'], 'r')
#     plt.title('total_loss_rpn')
#
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
#     plt.title('elapsed_time')
#
#     plt.show()
#
#     return record_df


def init_models():
    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)

    return model_rpn, class_mapping, model_classifier_only


def draw_box_on_images():
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    if from_csv:
        imgs_record_df = pd.read_csv(imgs_record_path)
        last_row = imgs_record_df.tail(1)
        test_imgs_temp = ast.literal_eval(last_row['train'].tolist()[0])
        test_imgs = []
        for img_dict in test_imgs_temp:
            test_imgs.append(img_dict['filepath'])
    else:
        test_imgs_temp = os.listdir(data_test_path)
        test_imgs = []
        for img_name in test_imgs_temp:
            test_imgs.append(os.path.join(data_test_path, img_name))

    # imgs_path = []
    # for i in range(10):
    #     idx = np.random.randint(len(test_imgs))
    #     imgs_path.append(test_imgs[idx])
    #
    # all_imgs = []
    #
    # classes = {}

    bbox_threshold = 0.7

    # for idx, img_name in enumerate(imgs_path):
    length = len(test_imgs)
    print(length)
    for idx, filepath in enumerate(test_imgs):
        print("Progression : " + str(idx + 1) + "/" + str(length))
        if not filepath.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(filepath)
        st = time.time()

        img = cv2.imread(filepath)

        X, ratio = format_img(img, C)

        X = np.transpose(X, (0, 2, 3, 1))

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

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

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
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
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              4)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 1)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        plt.figure(figsize=(10, 10))
        plt.grid()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        print(class_mapping)


def accuracy():
    if from_csv:
        imgs_record_df = pd.read_csv(imgs_record_path)
        last_row = imgs_record_df.tail(1)
        test_imgs = ast.literal_eval(last_row['train'].tolist()[0])
    else:
        test_imgs, _, _ = get_data(test_path, data_test_path)

    T = {}
    P = {}
    mAPs = []
    mRecalls = []
    mAccs = []
    for idx, img_data in enumerate(test_imgs):
        print('{}/{}'.format(idx, len(test_imgs)))
        st = time.time()
        filepath = img_data['filepath']

        img = cv2.imread(filepath)

        X, fx, fy = format_img_map(img, C)

        # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
        X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

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

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate all classes' bboxes coordinates on resized image (300, 400)
            # Drop 'bg' classes bboxes
            for ii in range(P_cls.shape[1]):

                # If class name is 'bg', continue
                if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # Get class name
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

            # Apply non-max-suppression on final bboxes to get the output bounding boxe
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                all_dets.append(det)

        print('Elapsed time = {}'.format(time.time() - st))
        t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])
        all_aps = []
        all_recalls = []
        all_accs = []
        for key in T.keys():
            ap = average_precision_score(T[key], P[key])
            recall = recall_score(T[key], [round(num) for num in P[key]])
            acc = accuracy_score(T[key], [round(num) for num in P[key]])
            print('{} AP: {}'.format(key, ap))
            print('{} Recall: {}'.format(key, recall))
            print('{} Acc: {}'.format(key, acc))
            all_aps.append(ap)
            all_recalls.append(recall)
            all_accs.append(acc)
        print('mAP = {}'.format(np.mean(np.array(all_aps))))
        print('mRecall = {}'.format(np.mean(np.array(all_recalls))))
        print('mAcc = {}'.format(np.mean(np.array(all_accs))))
        mAPs.append(np.mean(np.array(all_aps)))
        mRecalls.append(np.mean(np.array(all_recalls)))
        mAccs.append(np.mean(np.array(all_accs)))

    print()
    print('mean average precision:', np.nanmean(np.array(mAPs)))
    print('mean average recall:', np.nanmean(np.array(mRecalls)))
    print('mean average accuracy:', np.nanmean(np.array(mAccs)))

    mAP = [mAP for mAP in mAPs if str(mAP) != 'nan']
    mRecall = [mRecall for mRecall in mRecalls if str(mRecall) != 'nan']
    mAcc = [mAcc for mAcc in mAccs if str(mAcc) != 'nan']
    mean_average_prec = round(np.nanmean(np.array(mAP)), 3)
    mean_average_recall = round(np.nanmean(np.array(mRecall)), 3)
    mean_average_acc = round(np.nanmean(np.array(mAcc)), 3)
    print('The mean average precision is %0.3f' % (mean_average_prec))
    print('The mean average recall is %0.3f' % (mean_average_recall))
    print('The mean average accuracy is %0.3f' % (mean_average_acc))

    # record_df.loc[len(record_df)-1, 'mAP'] = mean_average_prec
    # record_df.to_csv(C.record_path, index=0)
    # print('Save mAP to {}'.format(C.record_path))


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Use Faster R-CNN to detect insects.')
    parser.add_argument('--model_name', required=False,
                        metavar="name_of_your_model", default='model',
                        help='Name of the model being tested')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/insects/test/dataset/", default='./data',
                        help='Directory of the Insects test dataset')
    parser.add_argument('--from_csv', required=False, default='True',
                        metavar="True/False",
                        help='True if loading test images from csv file')
    parser.add_argument('--annotations', required=False, default='./data/data_annotations.txt',
                        metavar="/path/to/insects/dataset/annotations/file.txt",
                        help='Annotation file for the provided dataset')
    parser.add_argument('--show_images', required=False, default=False,
                        metavar="True/False",
                        help="True if you want to show the images after detection, False otherwise")
    parser.add_argument('--use_gpu', required=False, default="True",
                        metavar="True/False",
                        help="True if you want to run the training on a gpu, False otherwise")
    args = parser.parse_args()

    use_gpu = eval(args.use_gpu)
    from_csv = eval(args.from_csv)

    if use_gpu is True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        session = tf.compat.v1.InteractiveSession(config=config)

    test_path = args.annotations  # Test data (annotation file)
    data_test_path = args.dataset

    imgs_record_path = "./logs/{} - imgs.csv".format(args.model_name)

    output_results_filename = "./results/{}".format(args.model_name)
    if not os.path.exists(output_results_filename):
        os.mkdir(output_results_filename)
    # TODO: output results of accuracy in file
    # TODO: parametrer le programme pour afficher les images ou calculer les metrics

    config_output_filename = "./config/{}.pickle".format(args.model_name)

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    C.model_path = "./model/{}.hdf5".format(args.model_name)  # UPDATE WEIGHTS PATH HERE !!!!!!!!

    # record_df = plot_some_graphs(C)
    model_rpn, class_mapping, model_classifier_only = init_models()
    if args.show_images:
        draw_box_on_images()
    accuracy()
