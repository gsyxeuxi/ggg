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


def hough_circle_radius(image, radius_range, threshold):
    height, width = image.shape
    max_radius = max(radius_range)
    
    accumulator = np.zeros((height, width, max_radius+1), dtype=np.uint32)
    
    edge_pixels = np.argwhere(image > 0)
    print(edge_pixels)
    
    for x, y in edge_pixels:
        for radius in radius_range:
            for theta in range(360):
                a = int(x - radius * np.cos(np.deg2rad(theta)))
                b = int(y + radius * np.sin(np.deg2rad(theta)))
                
                if 0 <= a < width and 0 <= b < height:
                    accumulator[b, a, radius] += 1
    detected_circles = np.argwhere(accumulator >= threshold)
    print(detected_circles)

    for i in detected_circles:
        center = (i[0], i[1])
        print(center)
        cv.circle(image, center, 2, (0,100,100), 3)
        raidus = i[2]
        # print(raidus)
        cv.circle(image, center, raidus, (255,0,255), 2)
    
    return image, i

