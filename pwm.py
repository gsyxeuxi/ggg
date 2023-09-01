import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(32,GPIO.OUT)
# GPIO.setup(32,GPIO.OUT)
GPIO.setup(33,GPIO.OUT)

p1 = GPIO.PWM(32, 1000)
p1.start(100)

# time.sleep(2) 
p2 = GPIO.PWM(33, 1000)
p2.start(50)
# time.sleep(20) 
try:
    while 1:
        for dc in range(0,101,5):
            p1.ChangeDutyCycle(dc)
            print("1")
            time.sleep(0.1)
        # for dc in range(0,101,5):
        #     p2.ChangeDutyCycle(dc)
        #     print("2")
        #     time.sleep(0.1) 
except KeyboardInterrupt:
    pass


p1.stop()
p2.stop()
GPIO.cleanup()