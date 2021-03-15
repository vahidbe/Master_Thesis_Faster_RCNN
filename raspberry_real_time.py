from real_time_libraries import *
from multiprocessing_detection_libraries import *

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


    bbox_threshold = float(args.bbox_threshold)
    output_results_filename = "./results/{}_{}".format(args.model_name, get_timestamp())
    if not os.path.exists(output_results_filename):
        os.mkdir(output_results_filename)
    record_path = os.path.join(output_results_filename,
                               "detections_{}.csv".format(args.model_name, get_timestamp()))

    if demo:
        run_demo(C, bbox_threshold)
    else:
        frame_queue = Queue()
        flag_queue = Queue()
        p_detection = Process(target=get_imgs, args=(1, 0.3, 5000, frame_queue, flag_queue))
        p_processing = Process(target=detection, args=(bbox_threshold, C, record_path, frame_queue, flag_queue))
        p_detection.start()
        p_processing.start()
        try:
            p_detection.join()
            p_processing.join()
        except KeyboardInterrupt:
            print("Stopping subprocesses")
            p_detection.terminate()
            p_processing.terminate()
            print("Exiting")


