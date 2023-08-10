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


def hough_circle_radius(image, radius_range_min, radius_range_max, threshold):
    height, width = image.shape
    accumulator = np.zeros((height, width, radius_range_max+1), dtype=np.uint32)
    
    edge_pixels = np.argwhere(image > 0)
    print(edge_pixels)
    
    for x, y in edge_pixels:
        for radius in range(radius_range_min, radius_range_max+1):
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


def hough_circle_gradient(image, radius_range_min, radius_range_max, threshold):
    
    gradient_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude and direction
    # gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_angle_deg = np.degrees(gradient_direction)

    edge_pixels = np.argwhere(image > 0)
    accumulator = np.zeros((540, 540), dtype=np.uint32)
    for y, x in edge_pixels:
        if x >= 100 and x <= 400 and y >= 100 and y <= 400:
            angle = gradient_angle_deg[y, x]
            for radius in range(radius_range_min, radius_range_max+1):
                dx = np.cos(np.radians(angle)) * radius
                dy = np.sin(np.radians(angle)) * radius
                x1 = int(x - dx)
                y1 = int(y - dy)
                if 0 <= x1 < 540 and 0 <= y1 < 540:
                    accumulator[x1, y1] += 1
                x2 = int(x + dx)
                y2 = int(y + dy)
                if 0 <= x2 < 540 and 0 <= y2 < 540:
                    accumulator[x2, y2] += 1
    detected_circles = np.argwhere(accumulator >= threshold)
    print(len(detected_circles))
    for i in detected_circles:
        center = (i[0], i[1])
    # print(center)
        cv.circle(image, center, 1, (255,0,255), 1)

    return image, i