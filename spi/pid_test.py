import time
import ADS1263
import Jetson.GPIO as GPIO

REF = 5.03 

angle = [0.0, 0.0]
angle_diff = [0.0, 0.0]
angle_diff_sum = [0.0, 0.0]
angle_diff_last = [0.0, 0.0]
kp = 0.0001
ki = 0.0001
kd = 0.0001

try:
    # set up ADC
    ADC = ADS1263.ADS1263()
    # choose the rate here (100Hz)
    if (ADC.ADS1263_init_ADC1('ADS1263_100SPS') == -1):
        exit()
    ADC.ADS1263_SetMode(0) # 0 is singleChannel, 1 is diffChannel
    channelList = [0, 1]  # The channel must be less than 10

    # set up PWM
    GPIO.setmode(GPIO.BOARD)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(32, GPIO.OUT, initial=GPIO.HIGH)
    p1 = GPIO.PWM(32, 1000)
    GPIO.setup(33, GPIO.OUT, initial=GPIO.HIGH)
    p2 = GPIO.PWM(33, 1000)
    val = [0, 0]
    p1.start(val[0])
    p2.start(val[1])

    angle_set = [0.0, 0.0]
    angle_set[0] = float(input("Angle 1 ="))
    angle_set[1] = float(input("Angle 2 ="))

    while(1):
        ADC_Value = ADC.ADS1263_GetAll(channelList)    # get ADC1 value
        for i in channelList:
            if(ADC_Value[i]>>31 ==1):       #received negativ value
                angle[i] = -round(((REF*2 - ADC_Value[i] * REF / 0x80000000) - 2.5) * 3, 2)
                print('angle', str(i+1), ' = ', angle[i], '°', sep="")
            else:       #received positiv value
                #change receive data (0.5V, 4.5V) to angle (-6°, 6°)
                receive_data = (ADC_Value[i] * REF / 0x7fffffff)
                angle[i] = round((receive_data - 2.5) * 3, 2)   # 32bit
                print('angle', str(i+1), ' = ', angle[i], '°', sep="")
            for i in channelList:
                print("\33[2A")
            
            angle_diff[i] = angle_set[i] - angle[i]
            angle_diff_sum[i] += angle_diff[i]
            val[i] = kp * angle_diff[i] + ki * angle_diff_sum[i] + kd * (angle_diff_last[i] - angle_diff[i])
            if val[i] > 100:
                val[i] = 100
            if val[i] < 0:
                val[i] = 0
            angle_diff_last[i] = angle_diff[i]
            if i == 0:
                p1.ChangeDutyCycle(val[i])
            else:
                p2.ChangeDutyCycle(val[i])

except IOError as e:
    print(e)
   
except KeyboardInterrupt:
    print("ctrl + c:")
    print("Program end")
    p1.stop()
    p2.stop()
    ADC.ADS1263_Exit()
    GPIO.cleanup()
    exit()