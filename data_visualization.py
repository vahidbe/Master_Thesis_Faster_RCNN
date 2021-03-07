import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import os
import pandas as pd
import ast



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
    plt.savefig(os.path.join('./other/histograms', title))

def get_box_image_ratio(input_path, set):

    global imgs_temp
    ratio_surface = {}

    ratio_surface["abeille_mellifere"] = []
    ratio_surface["boudon_des_arbres"] = []
    ratio_surface["anthophore_plumeuse"] = []
    ratio_surface["bourdon_des_jardins"] = []

    ratio_width_height = {}

    ratio_width_height["abeille_mellifere"] = []
    ratio_width_height["boudon_des_arbres"] = []
    ratio_width_height["anthophore_plumeuse"] = []
    ratio_width_height["bourdon_des_jardins"] = []

    rgb_intensity = {}

    rgb_intensity["abeille_mellifere"] = []
    rgb_intensity["boudon_des_arbres"] = []
    rgb_intensity["anthophore_plumeuse"] = []
    rgb_intensity["bourdon_des_jardins"] = []

    gray_intensity = {}

    gray_intensity["abeille_mellifere"] = []
    gray_intensity["boudon_des_arbres"] = []
    gray_intensity["anthophore_plumeuse"] = []
    gray_intensity["bourdon_des_jardins"] = []

    i = 1

    imgs_record_df = pd.read_csv(input_path)
    last_row = imgs_record_df.tail(1)
    if(set == 'train'):
        imgs_temp = ast.literal_eval(last_row['test'].tolist()[0])
    elif(set == 'test'):
        imgs_temp = ast.literal_eval(last_row['train'].tolist()[0])

    for img_dict in imgs_temp:
        sys.stdout.write('\r' + 'idx=' + str(i))
        i += 1

        thepath = img_dict['filepath']
        bboxes = img_dict['bboxes'][0]
        x1, y1, x2, y2, class_name = bboxes['x1'], bboxes['y1'], bboxes['x2'], bboxes['y2'], bboxes['class']
        hbox = np.abs(int(y1) - int(y2))
        wbox = np.abs(int(x1) - int(x2))
        box_surface = hbox * wbox

        # thepath = os.path.join(data_path, filename)
        im = cv2.imread(thepath)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mean_rgb_intensity = np.mean(im)
        mean_gray_intensity = np.mean(im_gray)
        h, w, c = im.shape
        surface = h * w
        ratio_surface[class_name].append(box_surface / surface)
        ratio_width_height[class_name].append(wbox / hbox)
        rgb_intensity[class_name].append(mean_rgb_intensity)
        gray_intensity[class_name].append(mean_gray_intensity)

    # with open(input_path, 'r') as f:
    #
    #     print('Parsing annotation files')
    #
    #     for line in f:
    #
    #         sys.stdout.write('\r' + 'idx=' + str(i))
    #         i += 1
    #
    #         line_split = line.strip().split(',')
    #
    #         (filename, x1, y1, x2, y2, class_name) = line_split
    #         hbox = np.abs(int(y1)-int(y2))
    #         wbox = np.abs(int(x1)-int(x2))
    #         box_surface = hbox * wbox
    #
    #         thepath = os.path.join(data_path, filename)
    #         im = cv2.imread(thepath)
    #         im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #         mean_rgb_intensity = np.mean(im)
    #         mean_gray_intensity = np.mean(im_gray)
    #         h, w, c = im.shape
    #         surface = h * w
    #         ratio_surface[class_name].append(box_surface/surface)
    #         ratio_width_height[class_name].append(wbox/hbox)
    #         rgb_intensity[class_name].append(mean_rgb_intensity)
    #         gray_intensity[class_name].append(mean_gray_intensity)

    return ratio_surface, ratio_width_height, rgb_intensity, gray_intensity


if __name__ == '__main__':
    # train_path = './data/data_annotations.txt'  # Training data (annotation file)
    # data_path = './data'
    #
    # test_path = './data_test/data_annotations.txt'  # Training data (annotation file)
    # data_test_path = './data_test'

    csv_file = './logs/model6000_val_epoch - imgs.csv'

    ratio_surface, ratio_width_height, rgb_intensity, gray_intensity = get_box_image_ratio(csv_file, 'train')
    ratio_surface_test, ratio_width_height_test, rgb_intensity_test, gray_intensity_test = get_box_image_ratio(csv_file, 'test')


    for class_name in ratio_surface.keys():
        display_histogram(ratio_surface[class_name], '[Train] Ratio box on image - ' + class_name, type='train')

    for class_name in ratio_surface_test.keys():
        display_histogram(ratio_surface_test[class_name], '[Test] Ratio box on image - ' + class_name, type='test')

    for class_name in ratio_width_height.keys():
        display_histogram(ratio_width_height[class_name], '[Train] Ratio width on height - ' + class_name, type='train')

    for class_name in ratio_width_height_test.keys():
        display_histogram(ratio_width_height_test[class_name], '[Test] Ratio width on height - ' + class_name, type='test')

    for class_name in rgb_intensity.keys():
        display_histogram(rgb_intensity[class_name], '[Train] RGB Intensity - ' + class_name, type='train')

    for class_name in rgb_intensity_test.keys():
        display_histogram(rgb_intensity_test[class_name], '[Test] RGB Intensity - ' + class_name, type='test')

    for class_name in gray_intensity.keys():
        display_histogram(gray_intensity[class_name], '[Train] Gray Intensity - ' + class_name, type='train')

    for class_name in gray_intensity_test.keys():
        display_histogram(gray_intensity_test[class_name], '[Test] Gray Intensity - ' + class_name, type='test')

