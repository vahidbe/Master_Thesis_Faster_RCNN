import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import os



def display_histogram(hist, title, type):
    hist, bin_edges = np.histogram(hist)
    plt.figure(figsize=[10, 8])

    if type == 'train':
        col = '#0504aa'
    elif type == 'test':
        col = '#d533df'
    else:
        col = '#5cb226'

    plt.bar(bin_edges[:-1], hist, width=0.5, color=col, alpha=None)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.title(title, fontsize=15)
    plt.show()

def get_box_image_ratio(input_path, data_path):

    ratio_dict = {}
    ratio_dict["abeille_mellifere"] = []
    ratio_dict["boudon_des_arbres"] = []
    ratio_dict["anthophore_plumeuse"] = []
    ratio_dict["bourdon_des_jardins"] = []

    i = 1

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:

            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1

            line_split = line.strip().split(',')

            (filename, x1, y1, x2, y2, class_name) = line_split
            hbox = np.abs(int(y1)-int(y2))
            wbox = np.abs(int(x1)-int(x2))
            box_surface = hbox * wbox

            thepath = os.path.join(data_path, filename)
            im = cv2.imread(thepath)
            h, w, c = im.shape
            surface = h * w
            ratio_dict[class_name].append(box_surface/surface)

    return ratio_dict


if __name__ == '__main__':
    train_path = './data/valid_data_annotations.txt'  # Training data (annotation file)
    data_path = './data'

    test_path = './data_test/data_annotations.txt'  # Training data (annotation file)
    data_test_path = './data_test'

    ratio_dict = get_box_image_ratio(train_path, data_path)
    for class_name in ratio_dict.keys():
        display_histogram(ratio_dict[class_name], '[Train] Ratio box on image : ' + class_name, type='train')

    ratio_dict_test = get_box_image_ratio(test_path, data_test_path)
    for class_name in ratio_dict_test.keys():
        display_histogram(ratio_dict_test[class_name], '[Test] Ratio box on image : ' + class_name, type='test')

