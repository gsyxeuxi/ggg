import multiprocessing
import pypylon.pylon as py
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
from circles_det import detect_circles

# num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
num_processes = 6
dp = 1  # Inverse ratio of the accumulator resolution to the image resolution
min_dist = 50  # Minimum distance between the centers of the detected circles
param1 = 100   # Upper threshold for the internal Canny edge detector
param2 = 36   # Threshold for center detection
min_Radius = 26  # Minimum circle radius
max_Radius = 32  # Maximum circle radius


tlf = py.TlFactory.GetInstance()
device = tlf.CreateFirstDevice()
cam = py.InstantCamera(device)
cam.Open()

#reset the camera
cam.UserSetSelector = "UserSet2"
#cam.UserSetSelector = "Default"
cam.UserSetLoad.Execute()

cam.AcquisitionFrameRateEnable.SetValue(True)
cam.AcquisitionFrameRate.SetValue(100)

cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)

previous_time = time.time()

while cam.IsGrabbing():
    grabResult = cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
    print(grabResult.GetNumberOfSkippedImages())

    if grabResult.GrabSucceeded():

        img = grabResult.Array
        img = cv.medianBlur(img, 7)
        
        # Split the image into chunks if you want to use multiprocessing with a single image.
        # If you have multiple images, you can process them individually with multiprocessing.

        chunks = np.array_split(img, num_processes)  # Split the image into chunks

        # Create a manager to store the detected circles across processes
        manager = multiprocessing.Manager()
        circles_output = manager.list()
                                                            
        # Create a list to store the processes
        processes = []

        # Create and start processes for each image chunk
        for i in range(num_processes):
            process = multiprocessing.Process(target=detect_circles,
                                            args=(chunks[i], circles_output, dp, min_dist, param1, param2, min_Radius, max_Radius))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        
        # detect_circles(img, cv.HOUGH_GRADIENT, dp=1, min_dist=50, param1=100, param2=36, min_Radius=26, max_Radius=32)
        current_time = time.time()
        latency = current_time - previous_time
        previous_time = current_time
        print(latency)
        
        cv.namedWindow('title', cv.WINDOW_NORMAL)
        cv.imshow('title', img)
        k = cv.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

cam.StopGrabbing()

cv.destroyAllWindows()


cam.Close()



# Combine the circles from all chunks
# all_circles = list(circles_output)
