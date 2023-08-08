import pypylon.pylon as py
import numpy as np
import cv2 as cv
import vpi
import time
import sys
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
            output = input.image_stats(flags=vpi.ImageStatistics.COVARIANCE)
            
        stats = output.cpu().view(np.recarray)[0]
        print(stats.covariance[0][0])

        current_time = time.time()
        latency = round(100 * (current_time - previous_time), 2)
        previous_time = current_time
        print(str('latency is:'), latency, str('ms'))

        cv.namedWindow('title', cv.WINDOW_NORMAL)
        cv.imshow('title', img)
        k = cv.waitKey(1)
        if k == 27:
            break
    grabResult.Release()

cam.StopGrabbing()
cv.destroyAllWindows()
cam.Close()