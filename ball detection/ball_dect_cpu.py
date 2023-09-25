import pypylon.pylon as py
import numpy as np
import cv2 as cv
import time
from circles_det import detect_circles_cpu
from coord_trans import coordinate_transform

def main():
    x0 = 268
    y0 = 260
    x1 = 156
    y1 = 21
    x2 = 369
    y2 = 24

    tlf = py.TlFactory.GetInstance()
    device = tlf.CreateFirstDevice()
    cam = py.InstantCamera(device)
    cam.Open()

    #reset the camera
    cam.UserSetSelector = "UserSet2"
    # cam.UserSetSelector = "Default"
    cam.UserSetLoad.Execute()
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(100)
    # print(cam.ExposureTime.Value)
    # print(cam.AcquisitionFrameRate.Value)

    cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)

    previous_time = time.time()


    while cam.IsGrabbing():
        grabResult = cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        print(str('Number of skipped images:'), grabResult.GetNumberOfSkippedImages())
        
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            img = cv.GaussianBlur(img,(3, 3),0)
            dectect_back = detect_circles_cpu(img, cv.HOUGH_GRADIENT, dp=1, min_dist=50, param1=100, param2=26, min_Radius=26, max_Radius=32)
            x = dectect_back[1][0]
            y = dectect_back[1][1]

            #coordinate transform
            inver_matrix = coordinate_transform(x1, y1, x2, y2, x0, y0)
            real_pos = np.round(np.dot(inver_matrix, np.array(([x],[y],[1]))))
            real_pos_x = real_pos[0][0] - 1
            real_pos_y = real_pos[1][0]
            print(real_pos_x, real_pos_y)

            current_time = time.time()
            latency = round(1000 * (current_time - previous_time), 2)
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


if __name__ == '__main__':
    main()