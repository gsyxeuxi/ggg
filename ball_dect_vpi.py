import pypylon.pylon as py
import numpy as np
import cv2 as cv
import vpi
import time
import sys
import matplotlib.pyplot as plt
from circles_det import circle_hough_transform
from coord_trans import coordinate_transform

tlf = py.TlFactory.GetInstance()
device = tlf.CreateFirstDevice()
cam = py.InstantCamera(device)
cam.Open()

#reset the camera
cam.UserSetSelector = "UserSet2"
cam.UserSetLoad.Execute()
cam.AcquisitionFrameRateEnable.SetValue(True)
cam.AcquisitionFrameRate.SetValue(200)

cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)

previous_time = time.time()

while cam.IsGrabbing():
    grabResult = cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
    print(str('Number of skipped images:'), grabResult.GetNumberOfSkippedImages())
    
    if grabResult.GrabSucceeded():
        img = grabResult.Array
        try:
            input = vpi.asimage(img)
        except IOError:
            sys.exit("Input file not found")
        except:
            sys.exit("Error with input file")
        with vpi.Backend.CUDA:
             output = input.canny(thresh_strong=300, thresh_weak=100, edge_value=255, nonedge_value=0, norm=vpi.Norm.L2)
        # inver_matrix = coordinate_transform(x1, y1, x2, y2, x0, y0)
        # real_pos = np.round(np.dot(inver_matrix, np.array(([x],[y],[1]))))
        # print(real_pos[0][0], real_pos[1][0])
        with output.rlock_cpu() as outData:
            # circle = circle_hough_transform(outData, (26,32), 10)
            cv.namedWindow('title', cv.WINDOW_NORMAL)
            cv.imshow('title', outData)
        
        current_time = time.time()
        latency = round(100 * (current_time - previous_time), 2)
        previous_time = current_time
        print(str('latency is:'), latency, str('ms'))

        k = cv.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

cam.StopGrabbing()
cv.destroyAllWindows()
cam.Close()