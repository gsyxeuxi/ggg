import time
import math
import ADS1263
import Jetson.GPIO as GPIO
import pypylon.pylon as py
import numpy as np
import cv2 as cv
from circles_det import detect_circles_cpu
from coord_trans import coordinate_transform
from multiprocessing import Process, Value


inver_matrix = coordinate_transform() * 400 / 540

def PIDPlate(angle_1, angle_2, pos_set_x, pos_set_y):
    REF = 5.03 
    angle = [0.0, 0.0]
    angle_diff = [0.0, 0.0]
    angle_diff_sum = [0.0, 0.0]
    angle_diff_last = [0.0, 0.0]
    angle_set = [0.0, 0.0]
    kp = 0.3
    ki = 0.07
    kd = 2.9
    # set up PWM
    GPIO.setmode(GPIO.BCM)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(12, GPIO.OUT, initial=GPIO.HIGH)
    p1 = GPIO.PWM(12, 1000)
    GPIO.setup(13, GPIO.OUT, initial=GPIO.HIGH)
    p2 = GPIO.PWM(13, 1000)
    val = [0, 0]
    p1.start(val[0])
    p2.start(val[1])
    try:
        # set up ADC
        ADC = ADS1263.ADS1263()
        # choose the rate here (100Hz)
        if (ADC.ADS1263_init_ADC1('ADS1263_100SPS') == -1):
            exit()
        ADC.ADS1263_SetMode(0) # 0 is singleChannel, 1 is diffChannel
        channelList = [0, 1]  # The channel must be less than 10
        previous_time = time.time()

        while(1):
            angle_set[0] = angle_1.value
            angle_set[1] = angle_2.value
            ADC_Value = ADC.ADS1263_GetAll(channelList)    # get ADC1 value
            for i in channelList:
                if(ADC_Value[i]>>31 ==1): #received negativ value, but potentiometer should not return negativ value
                    print('negativ potentiometer value received')
                    exit()
                else:       #potentiometer received positiv value
                    #change receive data in V to angle in °
                    receive_data = ADC_Value[i] * REF / 0x7fffffff
                    angle[i] = float('%.2f' %((receive_data - 2.51) * 2.91))   # 32bit
                    print('angle', str(i+1), ' = ', angle[i], '°', sep="")
                    # print("ADC1 IN%d = %lf" %(i, (ADC_Value[i] * REF / 0x7fffffff))) 
            
                angle_diff[i] = angle_set[i] - angle[i]
                angle_diff_sum[i] += angle_diff[i]
                val[i] = 100 - 20 * (2.5 + kp * angle_diff[i] + ki * angle_diff_sum[i] + kd * (angle_diff[i] - angle_diff_last[i]))
                if val[i] > 100:
                    val[i] = 100
                if val[i] < 0:
                    val[i] = 0
                angle_diff_last[i] = angle_diff[i]
                # print(val[i])
                if i == 0:
                    p1.ChangeDutyCycle(val[i])
                    # p1.ChangeDutyCycle(100)
                else:
                    p2.ChangeDutyCycle(val[i])
                    # p2.ChangeDutyCycle(100)
            for i in channelList:
                print("\33[2A")
            time.sleep(0.01)
            current_time = time.time()
            latency = round(1000 * (current_time - previous_time), 2)
            previous_time = current_time
            print(str('latency is:'), latency, str('ms'))
            
    except IOError as e:
        print(e)
    
    except KeyboardInterrupt:
        print("ctrl + c:")
        print("Program end")
        p1.stop()
        p2.stop()
        ADC.ADS1263_Exit()
        exit()


def PIDBall(angle_1, angle_2, pos_set_x, pos_set_y):
    angle = [0.0, 0.0]
    pos_diff = [0.0, 0.0]
    pos_diff_sum = [0.0, 0.0]
    pos_diff_last = [0.0, 0.0]
    kp = 0.001
    ki = 0.0005
    kd = 0.05
    
    tlf = py.TlFactory.GetInstance()
    device = tlf.CreateFirstDevice()
    cam = py.InstantCamera(device)
    cam.Open()
    #reset the camera
    cam.UserSetSelector = "UserSet2"
    cam.UserSetLoad.Execute()
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(100)
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
            real_pos = np.round(np.dot(inver_matrix, np.array(([x],[y],[1]))))
            real_pos_x = real_pos[0][0] - 1
            real_pos_y = real_pos[1][0]
            pos_diff[0] = pos_set_x.value - real_pos_x
            pos_diff[1] = pos_set_y.value - real_pos_y
            for i in range(2):
                pos_diff_sum[i] += pos_diff[i]
                angle[i] = kp * pos_diff[i] + ki * pos_diff_sum[i] + kd * (pos_diff[i] - pos_diff_last[i])
                if angle[i] > 6:
                    angle[i] = 6
                if angle[i] < -6:
                    angle[i] = -6
                pos_diff_last[i] = pos_diff[i]

            angle_1.value = angle[0]
            angle_2.value = angle[1]

            current_time = time.time()
            latency = round(1000 * (current_time - previous_time), 2)
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
    pos_set_x = Value('d', float(input("Pos x =")))
    pos_set_y = Value('d', float(input("Pos y =")))
    angle_1 = Value('d', 0.0)
    angle_2 = Value('d', 0.0)
    ball_process = Process(target=PIDBall, args=(angle_1, angle_2, pos_set_x, pos_set_y,))
    plate_process = Process(target=PIDPlate, args=(angle_1, angle_2, pos_set_x, pos_set_y,))
    ball_process.start()
    plate_process.start()
    ball_process.join()
    plate_process.join()
    # print("\33[1A")
    

if __name__ == '__main__':
    main()