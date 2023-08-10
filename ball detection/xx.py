import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

# Calculate the gradient in the x and y directions
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the gradient magnitude and direction
# gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)
gradient_angle_deg = np.degrees(gradient_direction)

edge_pixels = np.argwhere(image > 0)
accumulator = np.zeros((540, 540), dtype=np.uint32)

radius_min = 26
radius_max = 30

for y, x in edge_pixels:
    if x >= 100 and x <= 400 and y >= 100 and y <= 400:
        angle = gradient_angle_deg[y, x]
        for radius in range(radius_min, radius_max+1):
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
detected_circles = np.argwhere(accumulator >= 8)
print(len(detected_circles))
for i in detected_circles:
    center = (i[0], i[1])
    # print(center)
    cv2.circle(image, center, 1, (255,0,255), 1)

cv2.namedWindow('title', cv2.WINDOW_NORMAL)
cv2.imshow('title', image)
cv2.waitKey(100000)

# plt.show()
