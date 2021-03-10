from real_time_libraries import *


if __name__ == "__main__":
    import argparse
    import csv

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Use Faster R-CNN to detect insects.')
    parser.add_argument('--model_name', required=False,
                        metavar="name_of_your_model", default='model',
                        help='Name of the model being tested')
    parser.add_argument('--bbox_threshold', required=False, default=0.7,
                        metavar="Value from 0 to 1",
                        help="Model probability threshold to detect an object")
    parser.add_argument('--demo', required=False, default="False",
                        metavar="True/False",
                        help="True if you want to run the demo, False otherwise")
    parser.add_argument('--use_gpu', required=False, default="True",
                        metavar="True/False",
                        help="True if you want to run the training on a gpu, False otherwise")
    args = parser.parse_args()

    use_gpu = eval(args.use_gpu)
    demo = eval(args.demo)

    if use_gpu is True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        session = tf.compat.v1.InteractiveSession(config=config)

    config_output_filename = "./config/{}.pickle".format(args.model_name)

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    C.model_path = "./model/{}.hdf5".format(args.model_name)  # UPDATE WEIGHTS PATH HERE !!!!!!!!

    # record_df = plot_some_graphs(C)
    print("[INFO] loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    bbox_threshold = float(args.bbox_threshold)

    if demo:
        run_demo(model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
    else:
        output_results_filename = "./results/{}".format(args.model_name)
        if not os.path.exists(output_results_filename):
            os.mkdir(output_results_filename)
        record_path = os.path.join(output_results_filename,
                                   "detections_{}.csv".format(args.model_name, get_timestamp()))
        fieldnames = ['date', 'class', 'probability', 'x1', 'y1', 'x2', 'y2']
        detections_iterator = get_detections(model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
        while True:
            try:
                detection, probability, ((x1, y1), (x2, y2)) = next(detections_iterator)
                with open(record_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({'date': get_timestamp(), 'class': detection, 'probability': round(probability, 3),
                                     'x1': round(x1, 3), 'y1': round(y1, 3), 'x2': round(x2, 3), 'y2': round(y2, 3)})
            except StopIteration:
                exit(0)
