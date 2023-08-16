import time
import Jetson.GPIO as GPIO
import pypylon.pylon as py
import numpy as np
import cv2 as cv
from circles_det import detect_circles_cpu
from coord_trans import coordinate_transform
from multiprocessing import Process, Value, Array

# transform form pixel to mm
inver_matrix = coordinate_transform() * 400 / 540

def pwm(real_pos):
    # Pin Setup:
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(32, GPIO.OUT, initial=GPIO.HIGH)
    p1 = GPIO.PWM(32, 1000)
    GPIO.setup(33, GPIO.OUT, initial=GPIO.HIGH)
    p2 = GPIO.PWM(33, 1000)
    val_1 = 10
    val_2 = 90
    incr_1 = 10
    incr_2 = 10
    p1.start(val_1)
    p2.start(val_2)

    # pid parameter
    kp = 0.001
    ki = 0.0005
    kd = 0.0005



    print("PWM running. Press CTRL+C to exit.")
    try:
        while True:
            print('PWM read', real_pos[0], real_pos[1])
            time.sleep(0.5)
            if val_1 >= 100:
                incr_1 = -incr_1
            if val_2 >= 100:
                incr_2 = -incr_2
            if val_1 <= 0:
                incr_1 = -incr_1
            if val_2 <= 0:
                incr_2= -incr_2
            val_1 += incr_1
            val_2 += incr_2
            p1.ChangeDutyCycle(val_1)
            p2.ChangeDutyCycle(val_2)
    finally:
        p1.stop()
        p2.stop()
        GPIO.cleanup()



def ball_cv(real_pos):
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
        # print(str('Number of skipped images:'), grabResult.GetNumberOfSkippedImages())
        
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            img = cv.GaussianBlur(img,(5,5),0)
            dectect_back = detect_circles_cpu(img, cv.HOUGH_GRADIENT, dp=1, min_dist=50, param1=100, param2=36, min_Radius=26, max_Radius=32)
            x = dectect_back[1][0]
            y = dectect_back[1][1]

            #coordinate transform
            transed_pos = np.round(np.dot(inver_matrix, np.array(([x],[y],[1]))))
            real_pos[0] = transed_pos[0][0] - 1
            real_pos[1]= transed_pos[1][0]
            # print(real_pos[0], real_pos[1])

            current_time = time.time()
            latency = round(100 * (current_time - previous_time), 2)
            previous_time = current_time
            # print(str('latency is:'), latency, str('ms'))
            
            cv.namedWindow('title', cv.WINDOW_NORMAL)
            cv.imshow('title', img)
            k = cv.waitKey(1)
            if k == 27:
                break
        grabResult.Release()

    cam.StopGrabbing()
    cv.destroyAllWindows()
    cam.Close()


def main():
    real_pos = Array('d', range(2))
    cv_process = Process(target=ball_cv, args=(real_pos,))
    pwm_process = Process(target=pwm, args=(real_pos,))
    cv_process.start()
    pwm_process.start()
    cv_process.join()
    pwm_process.join()
    

if __name__ == '__main__':
    main()
    