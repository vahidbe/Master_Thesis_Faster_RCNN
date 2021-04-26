import itertools as it
import cv2
from matplotlib import pyplot as plt
import numpy as np


def adjust_gamma(image, gamma=2.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def equalize_hist(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return image

if __name__ =='__main__':
    # param = {
    #     'brightness_jitter': [True],
    #     'brightness_jitter_bound': [0.1, 0.2, 0.3, 0.4],
    #     'gamma_correction': [True],
    #     'gamma_value': np.linspace(1.0, 4.0, 7)
    # }
    #
    # paramNames = list(param.keys())
    # combinations = it.product(*(param[Name] for Name in paramNames))
    #
    #
    # paramNames = list(param.keys())
    # # combinations = it.product(*(param[Name] for Name in paramNames))
    #
    # combinations = [
    #     (True, 0.1, False, 1.0),
    #     (True, 0.2, False, 1.0),
    #     (True, 0.3, False, 1.0),
    #     (True, 0.4, False, 1.0),
    #     (False, 0.0, True, 1.0),
    #     (False, 0.0, True, 1.5),
    #     (False, 0.0, True, 2.0),
    #     (False, 0.0, True, 2.5),
    #     (False, 0.0, True, 3.0),
    #     (False, 0.0, True, 3.5),
    #     (False, 0.0, True, 4.0),
    # ]
    #
    # for param in combinations:
    #     print(param)
    #     print(type(param))
    #
    img = cv2.imread('./11-04/processed_22-04-2021_09-29-59.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('image', img.copy())
    img2 = equalize_hist(img)
    cv2.imwrite('./11-04/corrected2.jpg', img2)
    cv2.imshow('image2', img2)
    cv2.waitKey(0)

    # for g in np.linspace(1.5, 4.0, 6):
    #     img2 = adjust_gamma(img, gamma=g)
    #     cv2.imshow('image', img2)
    #     cv2.waitKey(0)

    cv2.destroyAllWindows()