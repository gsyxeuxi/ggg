import cv2 as cv
import numpy as np


def detect_circles_cpu(image_chunk, circles_output, min_Radius, max_Radius, dp, min_dist, param1, param2):
    circles = cv.HoughCircles(image_chunk, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2,
                              minRadius=min_Radius, maxRadius=max_Radius)
    if circles is not None:
        # circles = np.uint16(np.around(circles))
        circles = np.uint16(circles)

        # for i in circles[0,:]
        i = circles [0,0]
        center = (i[0], i[1])
        print(center)
        cv.circle(image_chunk, center, 2, (0,100,100), 3)
        raidus = i[2]
        # print(raidus)
        cv.circle(image_chunk, center, raidus, (255,0,255), 2)
    
    else:
        i = [0, 0, 0]

    return image_chunk, i


def detect_circles_gpu(image_chunk, circles_output, min_Radius, max_Radius, dp, min_dist, param1, param2):
    circles = cv.cuda.HoughCirclesDetector(image_chunk, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2,
                              minRadius=min_Radius, maxRadius=max_Radius)
    if circles is not None:
        # circles = np.uint16(np.around(circles))
        circles = np.uint16(circles)

        # for i in circles[0,:]
        i = circles [0,0]
        center = (i[0], i[1])
        print(center)
        cv.circle(image_chunk, center, 2, (0,100,100), 3)
        raidus = i[2]
        # print(raidus)
        cv.circle(image_chunk, center, raidus, (255,0,255), 2)
    
    else:
        i = [0, 0, 0]

    return image_chunk, i