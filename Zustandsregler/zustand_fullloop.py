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


def PIDPlate(angle_1, angle_2, pos_set_x, pos_set_y, vel_set_x, vel_set_y):
    REF = 5.03 
    angle = [0.0, 0.0]
    angle_diff = [0.0, 0.0]
    angle_diff_sum = [0.0, 0.0]
    angle_diff_last = [0.0, 0.0]
    angle_set = [0.0, 0.0]
    kp = 0.3
    ki = 0.07
    kd = 2.0
    # kp = 0.24
    # ki = 0.05
    # kd = 4.58
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
            t = time.time()
            angle_set[0] = angle_1.value
            angle_set[1] = angle_2.value
            # print('angle1 =', angle_set[0])
            # print('angle2 =',angle_set[1])
            # print("\33[2A")
            ADC_Value = ADC.ADS1263_GetAll(channelList)    # get ADC1 value
            for i in channelList:
                if(ADC_Value[i]>>31 ==1): #received negativ value, but potentiometer should not return negativ value
                    print('negativ potentiometer value received')
                    exit()
                else:       #potentiometer received positiv value
                    #change receive data in V to angle in °
                    receive_data = ADC_Value[i] * REF / 0x7fffffff
                    angle[i] = float('%.2f' %((receive_data - 2.51) * 2.91))   # 32bit
                    # print('angle', str(i+1), ' = ', angle[i], '°', sep="")

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
                # print(time.time() - t)
            # for i in channelList:
            #     print("\33[2A")
            # time.sleep(0.01)
            current_time = time.time()
            latency = round(1000 * (current_time - previous_time), 2)
            previous_time = current_time
            # print(str('latency is:'), latency, str('ms'))
            
    except IOError as e:
        print(e)
    
    except KeyboardInterrupt:
        print("ctrl + c:")
        print("Program end")
        p1.stop()
        p2.stop()
        ADC.ADS1263_Exit()
        exit()


def PIDBall(angle_1, angle_2, pos_set_x, pos_set_y, vel_set_x, vel_set_y):
    inver_matrix, transform_matrix = coordinate_transform()
    # pos_set_trans = np.round(np.dot(inver_matrix, np.array(([pos_set_x.value],[pos_set_y.value],[1]))))
    # pos_set_trans_x = int(pos_set_trans[0][0]) - 1
    # pos_set_trans_y = int(pos_set_trans[1][0])
    angle = [0.0, 0.0]
    pos_diff = [0.0, 0.0]
    pos_diff_last = [0.0, 0.0]
    pos_last_x = 0
    pos_last_y = 0
    vel_diff = np.zeros(2)
    # c0 = -0.03
    # c1 = -0.009
    c0 = -0.02
    c1 = -0.01
    latency = 1000/60
    
    tlf = py.TlFactory.GetInstance()
    device = tlf.CreateFirstDevice()
    cam = py.InstantCamera(device)
    cam.Open()
    #reset the camera
    cam.UserSetSelector = "UserSet2"
    cam.UserSetLoad.Execute()
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(60)
    cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    previous_time = time.time()
    while cam.IsGrabbing():
        grabResult = cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        # print(str('Number of skipped images:'), grabResult.GetNumberOfSkippedImages())
        if grabResult.GrabSucceeded():
            pos_set_trans = np.round(np.dot(transform_matrix, np.array(([pos_set_x.value*540/400],[pos_set_y.value*540/400],[1]))))
            pos_set_trans_x = int(pos_set_trans[0][0])
            pos_set_trans_y = int(pos_set_trans[1][0])
            img = grabResult.Array
            img = cv.GaussianBlur(img,(3,3),0)
            dectect_back = detect_circles_cpu(img, cv.HOUGH_GRADIENT, dp=1, min_dist=50, param1=100, param2=36, min_Radius=26, max_Radius=32)
            img= cv.drawMarker(img, (int(pos_set_trans_x), int(pos_set_trans_y)), (0, 0, 255), markerType=1)
            # img= cv.drawMarker(img, (int(pos_set_x.value), int(pos_set_y.value)), (0, 0, 255), markerType=1)
            x = dectect_back[1][0]
            y = dectect_back[1][1]
            #coordinate transform
            real_pos = np.round(np.dot(inver_matrix, np.array(([x],[y],[1]))))
            real_pos_x = real_pos[0][0] - 1
            real_pos_y = real_pos[1][0]
            pos_diff[0] = pos_set_x.value - real_pos_x
            pos_diff[1] = pos_set_y.value - real_pos_y
            vel_x = np.round((real_pos_x - pos_last_x) * 1000 / latency, 3) #dt = 1/60
            vel_y = np.round((real_pos_y - pos_last_y) * 1000 / latency, 3)
            print('velx is:', vel_x)
            print('velset_x is:', vel_set_x.value)
            vel_diff[0] = vel_set_x.value - vel_x
            vel_diff[1] = vel_set_y.value - vel_y
            # print(real_pos_x, real_pos_y)
            for i in range(2):
                angle[i] = round(c0 * pos_diff[i] + c1 * vel_diff[i], 3)
                if angle[i] > 6:
                    angle[i] = 6
                if angle[i] < -6:
                    angle[i] = -6
                # print('angle', str(i+1), ' = ', angle[i], '°', sep="")
                pos_diff_last[i] = pos_diff[i]
            # for i in range(2):
            #     print("\33[2A")
            
            pos_last_x = real_pos_x
            pos_last_y = real_pos_y
            # print(angle[0])
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

def trajectory(angle_1, angle_2, pos_set_x, pos_set_y, vel_set_x, vel_set_y): 
    #l: Half of the length of the diagonal of the square
    #p: Time peroide
    print('????????')
    l = 150
    p = 16
    a = l*np.sqrt(2)/(p/8)**2 #acceleration
    while True:
        t = time.time() % p  # Ensure that the trajectory repeats every p seconds
        if 0 <= t < p/8:
            pos_set_x.value = l - 0.5 * a * t**2/np.sqrt(2)
            pos_set_y.value = 0.5 * a * t**2/np.sqrt(2)
            vel_set_x.value = - a * t * 2/np.sqrt(2)
            vel_set_y.value = a * t * 2/np.sqrt(2)
        elif p/8 <= t < p/4:
            pos_set_x.value = 0.5 * a * (p/4-t)**2/np.sqrt(2)
            pos_set_y.value = l - 0.5 * a * (p/4-t)**2/np.sqrt(2)
            vel_set_x.value = - a * (p/4-t) * 2/np.sqrt(2)
            vel_set_y.value = a * (p/4-t) * 2/np.sqrt(2)
        elif p/4 <= t < 3*p/8:
            pos_set_x.value = -0.5 * a * (t-p/4)**2/np.sqrt(2)
            pos_set_y.value = l - 0.5 * a * (t-p/4)**2/np.sqrt(2)
            vel_set_x.value = - a * (t-p/4) * 2/np.sqrt(2)
            vel_set_y.value = - a * (t-p/4) * 2/np.sqrt(2)
        elif 3*p/8 <= t < p/2:
            pos_set_x.value = -l + 0.5 * a * (p/2-t)**2/np.sqrt(2)
            pos_set_y.value = 0.5 * a * (p/2-t)**2/np.sqrt(2)
            vel_set_x.value = - a * (p/2-t) * 2/np.sqrt(2)
            vel_set_y.value = - a * (p/2-t) * 2/np.sqrt(2)
        elif p/2 <= t < 5*p/8:
            pos_set_x.value = -l + 0.5 * a * (t-p/2)**2/np.sqrt(2)
            pos_set_y.value = -0.5 * a * (t-p/2)**2/np.sqrt(2)
            vel_set_x.value = a * (t-p/2) * 2/np.sqrt(2)
            vel_set_y.value = -a * (t-p/2) * 2/np.sqrt(2)
        elif 5*p/8 <= t < 3*p/4:
            pos_set_x.value = -0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            pos_set_y.value = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            vel_set_x.value = a * (3*p/4-t) * 2/np.sqrt(2)
            vel_set_y.value = -a * (3*p/4-t) * 2/np.sqrt(2)
        elif 3*p/4 <= t < 7*p/8:
            pos_set_x.value = 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            pos_set_y.value = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            vel_set_x.value = a * (t-3*p/4) * 2/np.sqrt(2)
            vel_set_y.value = a * (t-3*p/4) * 2/np.sqrt(2)
        else:
            pos_set_x.value = l - 0.5 * a * (t-p)**2/np.sqrt(2)
            pos_set_y.value = -0.5 * a * (t-p)**2/np.sqrt(2)
            vel_set_x.value = a * (p-t) * 2/np.sqrt(2)
            vel_set_y.value = a * (p-t) * 2/np.sqrt(2)

        # print(pos_set_x, pos_set_y, vel_set_x, vel_set_y)
        # return pos_set_x, pos_set_y, vel_set_x, vel_set_y


def main():
    pos_set_x = Value('d', 0)
    pos_set_y = Value('d', 0)
    vel_set_x = Value('d', 0.0)
    vel_set_y = Value('d', 0.0)
    angle_1 = Value('d', 0.0)
    angle_2 = Value('d', 0.0)
    ball_process = Process(target=PIDBall, args=(angle_1, angle_2, pos_set_x, pos_set_y, vel_set_x, vel_set_y,))
    plate_process = Process(target=PIDPlate, args=(angle_1, angle_2, pos_set_x, pos_set_y, vel_set_x, vel_set_y,))
    trajectory_process = Process(target=trajectory, args=(angle_1, angle_2, pos_set_x, pos_set_y, vel_set_x, vel_set_y,))
    trajectory_process.start()
    ball_process.start()
    plate_process.start()
    ball_process.join()
    plate_process.join()
    trajectory_process.join()


if __name__ == '__main__':
    main()