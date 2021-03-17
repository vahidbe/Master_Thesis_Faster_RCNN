import csv
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera


def init_session(use_gpu):
    from detection_libraries import tf
    from detection_libraries import os

    if use_gpu is True:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        session = tf.compat.v1.InteractiveSession(config=config)


def run_detection(fps, alpha, min_area, frame_queue, flag_queue):
    print("waiting for flag")
    flag = flag_queue.get(block=True, timeout=None)
    if flag == "ready":
        print("[INFO] detection_proc - Models ready: detection can start")
    else:
        print("[INFO] detection_proc - Error loading models: aborted")
        return

    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # initialize the first frame in the video stream
    avg = None
    # loop over the frames of the video
    while True:
        time.sleep(round(1 / fps))
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()

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
            print("[INFO] detection_proc - starting background model...")
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
            if cv2.contourArea(c) < min_area:
                continue
            print("[INFO] detection_proc - *** Movement detected ***")
            frame_queue.put((get_timestamp(), frame))
            break
        cv2.imshow("image", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def run_demo(C, bbox_threshold):
    import numpy as np
    from detection_libraries import init_models
    from detection_libraries import detect
    print("[INFO] loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    # initialize the video stream, allow the camera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    fps = FPS().start()
    while True:
        img = vs.read()
        imutils.resize(img, width=400)

        detect(img, model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
        cv2.imshow("Frame", img)

        key = cv2.waitKey(1000) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    print("[INFO] ending video stream...")
    vs.stop()


def run_processing(bbox_threshold, C, record_path, use_gpu, frame_queue, flag_queue):

    print("[INFO] processing_proc - initializing session...")
    import numpy as np
    from detection_libraries import init_models
    from detection_libraries import detect

    init_session(use_gpu)
    print("[INFO] processing_proc - loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    print("[INFO] processing_proc - done loading model")
    flag_queue.put("ready")
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    fieldnames = ['date', 'class', 'probability', 'x1', 'y1', 'x2', 'y2', 'temperature',
                  'pressure', 'wind', 'rain', 'weather description']
    while True:
        time.sleep(1)
        print("waiting for image")
        timestamp, img = frame_queue.get(block=True, timeout=None)
        print("image found")
        all_dets = detect(img, model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
        if not len(all_dets) == 0:
            for detected_class, probability, ((x1, y1), (x2, y2)) in all_dets:
                with open(record_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(
                        {'date': timestamp, 'class': detected_class, 'probability': round(probability, 3),
                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'temperature': 'empty',
                         'pressure': 'empty', 'wind': 'empty', 'rain': 'empty',
                         'weather description': 'empty'})

        cv2.imshow("image", img)
        key = cv2.waitKey(1000) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def get_timestamp():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
