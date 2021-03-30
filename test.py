import itertools as it
import cv2
from matplotlib import pyplot as plt


if __name__ =='__main__':
    img = cv2.imread('./data/abeille_mellifere0000.jpg')
    blur = cv2.GaussianBlur(img,(7,7), 0)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.imshow('image', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()