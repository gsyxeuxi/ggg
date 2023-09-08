import time
import math
import ADS1263
import Jetson.GPIO as GPIO

REF = 5.03
angle = [0.0, 0.0]
angle_diff = [0.0, 0.0]
angle_diff_sum = [0.0, 0.0]
angle_diff_last = [0.0, 0.0]
angle_set = [0.0, 0.0]
kp = 0.15
ki = 0.01
kd = 0.15

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
    if ADC.ADS1263_init_ADC1('ADS1263_100SPS') == -1:
        exit()
    ADC.ADS1263_SetMode(0)  # 0 is singleChannel, 1 is diffChannel
    channelList = [0, 1]  # The channel must be less than 10
    previous_time = time.time()

    with open('data.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        while 1:
            angle_set[0] = 5 * math.sin(2 * math.pi * previous_time / 8)  # T = 8s
            angle_set[1] = 5 * math.sin(2 * math.pi * previous_time / 8 + math.pi / 2)
            ADC_Value = ADC.ADS1263_GetAll(channelList)  # get ADC1 value
            for i in channelList:
                if ADC_Value[i] >> 31 == 1:  # received negative value, but potentiometer should not return negative
                    # value
                    print('negative potentiometer value received')
                    exit()
                else:  # potentiometer received positive value
                    # change receive data in V to angle in °
                    receive_data = ADC_Value[i] * REF / 0x7fffffff
                    angle[i] = float('%.2f' % ((receive_data - 2.51) * 2.91))  # 32bit
                    print('angle', str(i + 1), ' = ', angle[i], '°', sep="")
                    # print("ADC1 IN%d = %lf" %(i, (ADC_Value[i] * REF / 0x7fffffff)))

                angle_diff[i] = angle_set[i] - angle[i]
                angle_diff_sum[i] += angle_diff[i]
                val[i] = 100 - 20 * (2.5 + kp * angle_diff[i] + ki * angle_diff_sum[i] + kd * (
                            angle_diff[i] - angle_diff_last[i]))
                if val[i] > 100:
                    val[i] = 100
                if val[i] < 0:
                    val[i] = 0
                angle_diff_last[i] = angle_diff[i]
                print(val[i])
                if i == 0:
                    p1.ChangeDutyCycle(val[i])
                    # p1.ChangeDutyCycle(100)
                else:
                    p2.ChangeDutyCycle(val[i])
                    # p2.ChangeDutyCycle(100)
            for i in channelList:
                print("\33[2A")
            csv_writer.writerow([angle_set[0], angle_set[1], angle[0], angle[1]])
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
