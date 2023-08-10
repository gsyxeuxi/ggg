import cv2 as cv
import numpy as np
from circles_det import hough_circle_radius

image = cv.imread('1.png', cv.IMREAD_GRAYSCALE)
circle = hough_circle_radius(image, (26,28), 100)

cv.namedWindow('title', cv.WINDOW_NORMAL)
cv.imshow('title', image)
k = cv.waitKey(100000)

