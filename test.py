from real_time_libraries import *

def get_imgs():

    # import the necessary packages
    from imutils.video import VideoStream
    import argparse
    import datetime
    import imutils
    import time
    import cv2

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())
    # if the video argument is None, then we are reading from webcam
    if args.get("video", None) is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(args["video"])
    # initialize the first frame in the video stream
    firstFrame = None
    alpha = 0.1
    fps = 1
    avg = None
    # loop over the frames of the video
    while True:
        time.sleep(round(1/fps))
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        text = "Unoccupied"
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            break
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the first frame is None, initialize it
        if avg is None:
            print("[INFO] starting background model...")
            avg = gray.copy().astype("float")
            continue
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, alpha)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            # (x, y, w, h) = cv2.boundingRect(c)
            cv2.imshow("Security Feed", frame)
            yield frame
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # text = "Occupied"

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
    # cleanup the camera and close any open windows
    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import csv
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    session = tf.compat.v1.InteractiveSession(config=config)

    config_output_filename = "./config/{}.pickle".format("model6000_val_epoch")

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    C.model_path = "./model/{}.hdf5".format("model6000_val_epoch")  # UPDATE WEIGHTS PATH HERE !!!!!!!!

    # record_df = plot_some_graphs(C)
    print("[INFO] loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    bbox_threshold = float(0.7)

    it = get_imgs()
    from repeated_timer import RepeatedTimer

    # Initialize records
    output_results_filename = "./results/{}".format("model6000_val_epoch")
    if not os.path.exists(output_results_filename):
        os.mkdir(output_results_filename)
    record_path = os.path.join(output_results_filename,
                               "detections_{}_{}.csv".format("model6000_val_epoch", get_timestamp()))
    fieldnames = ['date',
                  'class', 'probability',
                  'x1', 'y1', 'x2', 'y2',
                  'temperature', 'pressure', 'wind', 'rain', 'weather description']
    with open(record_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({'date': 'date', 'class': 'class', 'probability': 'probability',
                         'x1': 'x1', 'y1': 'y1', 'x2': 'x2', 'y2': 'y2', 'temperature': 'temperature',
                         'pressure': 'pressure', 'wind': 'wind', 'rain': 'rain',
                         'weather description': 'weather description'})
    # Start logging detections
    detections_iterator = get_detections(model_rpn, model_classifier_only,
                                         C, class_mapping, bbox_threshold, class_to_color)
    while True:
        try:
            all_dets = detect(next(it), model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
            if not len(all_dets) == 0:
                for detection, probability, ((x1, y1), (x2, y2)) in all_dets:
                    with open(record_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow({'date': get_timestamp(), 'class': detection, 'probability': round(probability, 3),
                                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'temperature': "empty",
                                         'pressure': "empty", 'wind': "empty", 'rain': "empty",
                                         'weather description': "empty"})
        except StopIteration:
            exit(0)
