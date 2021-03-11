from real_time_libraries import *
import threading
import csv

def detectionThread(use_gpu, demo, bbox_threshold, model_name, iterator):

    if use_gpu is True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        session = tf.compat.v1.InteractiveSession(config=config)

    config_output_filename = "./config/{}.pickle".format(model_name)

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    C.model_path = "./model/{}.hdf5".format(model_name)  # UPDATE WEIGHTS PATH HERE !!!!!!!!

    # record_df = plot_some_graphs(C)
    print("[INFO] loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    output_results_filename = "./results/{}".format(model_name)
    if not os.path.exists(output_results_filename):
        os.mkdir(output_results_filename)
    record_path = os.path.join(output_results_filename,
                               "detections_{}.csv".format(model_name, get_timestamp()))
    fieldnames = ['date', 'class', 'probability', 'x1', 'y1', 'x2', 'y2']

    lock = threading.Lock()
    while True:
        try:
            with lock:
                img = next(iterator)
        except StopIteration:
            continue

        all_dets = detect(img, model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
        if not len(all_dets) == 0:
            for detection, probability, ((x1, y1), (x2, y2)) in all_dets:
                with open(record_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({'date': get_timestamp(), 'class': detection, 'probability': round(probability, 3),
                                     'x1': round(x1, 3), 'y1': round(y1, 3), 'x2': round(x2, 3), 'y2': round(y2, 3)})
