import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Check bounding boxes for insects.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/insects/dataset",
                        help='Path to insect dataset directory')
    parser.add_argument('--input', required=True,
                        metavar="/path/to/insects/bbox/annotation/file",
                        help='Bounding box annotation file')
    parser.add_argument('--output', required=True,
                        metavar="/path/to/output/bbox/file",
                        help="Bounding box annotation output file")
    parser.add_argument('--startfrom', required=False,
                        metavar="/path/to/starting/image",
                        help='Path to the first image')
    args = parser.parse_args()

    # Validate arguments
    input_file = args.input
    output_file = args.output
    dataset_dir = args.dataset
    start_from = args.startfrom
    has_started = False

    print("Dataset directory: ", dataset_dir)
    print("Input file: ", input_file)
    print("Output file: ", output_file)

    print("=== Press space to validate an image and escape (or close it) to invalidate it ===")

    files = os.listdir(dataset_dir)

    with open(input_file, 'r') as f:

        for line in f:

            line_split = line.strip().split(',')

            # Make sure the info saved in annotation file matching the format (path_filename, x1, y1, x2, y2, class_name)
            # Note:
            #	One path_filename might has several classes (class_name)
            #	x1, y1, x2, y2 are the pixel value of the origial image, not the ratio value
            #	(x1, y1) top left coordinates; (x2, y2) bottom right coordinates
            #   x1,y1-------------------
            #	|						|
            #	|						|
            #	|						|
            #	|						|
            #	---------------------x2,y2

            (filename, x1, y1, x2, y2, class_name) = line_split
            if start_from is not None and not has_started and not filename == start_from:
                continue

            has_started = True

            if not files.__contains__(filename):
                continue

            path = os.path.join(dataset_dir, filename)
            print(path)
            img = cv2.imread(path)

            while True:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
                cv2.imshow(filename, img)
                k = cv2.waitKey(1)
                if k%256 == 32:
                    # Spacebar
                    with open(output_file, 'a') as outputf:
                        outputf.write(str(filename) + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(
                            y2) + "," + str(class_name) + "\n")
                    break
                if k%256 == 27:
                    # Escape
                    break

            cv2.destroyAllWindows()
