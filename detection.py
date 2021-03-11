from real_time_libraries import *
import csv


def get_imgs(fps, alpha, min_area, queue):

    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    # initialize the first frame in the video stream
    avg = None
    # loop over the frames of the video
    while True:
        time.sleep(round(1/fps))
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
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
            if cv2.contourArea(c) < min_area:
                continue
            queue.put(frame)
            # cv2.imshow("image", queue.get())
            # key = cv2.waitKey(1) & 0xFF
            # # if the `q` key is pressed, break from the lop
            # if key == ord("q"):
            #     break


def run_demo(C, bbox_threshold):
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


def detection_proc(bbox_threshold, C, record_path, frame_queue):
    print("[INFO] loading model...")
    model_rpn, class_mapping, model_classifier_only = init_models(C)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    fieldnames = ['date', 'class', 'probability', 'x1', 'y1', 'x2', 'y2']
    while True:
        time.sleep(1)
        img = frame_queue.get(block=True, timeout=None)

        all_dets = detect(img, model_rpn, model_classifier_only, C, class_mapping, bbox_threshold, class_to_color)
        if not len(all_dets) == 0:
            for detection, probability, ((x1, y1), (x2, y2)) in all_dets:
                with open(record_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({'date': get_timestamp(), 'class': detection, 'probability': round(probability, 3),
                                     'x1': round(x1, 3), 'y1': round(y1, 3), 'x2': round(x2, 3), 'y2': round(y2, 3)})

        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
