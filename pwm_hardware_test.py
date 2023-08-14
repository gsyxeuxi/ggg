import time
import Jetson.GPIO as GPIO


def main():
    # Pin Setup:
    # Board pin-numbering scheme
    GPIO.setmode(GPIO.BOARD)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(32, GPIO.OUT, initial=GPIO.HIGH)
    p1 = GPIO.PWM(32, 1000)
    GPIO.setup(33, GPIO.OUT, initial=GPIO.HIGH)
    p2 = GPIO.PWM(33, 1000)
    val_1 = 0
    val_2 = 100
    incr_1 = 10
    incr_2 = 10
    p1.start(val_1)
    p2.start(val_2)

    print("PWM running. Press CTRL+C to exit.")
    try:
        while True:
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
    
    
    # try:
    #     while True:
    #         time.sleep(200)
    # finally:
    #     p1.stop()
    #     p2.stop()
    #     GPIO.cleanup()

    # while True:
    #     time.sleep(1)
    #     val2 = 100-val
    #     if val == 100:
    #         break
    #     val += incr
    #     p1.ChangeDutyCycle(val)
    #     p2.ChangeDutyCycle(val2)
    # p1.stop()
    # p2.stop()
    # GPIO.cleanup()

if __name__ == '__main__':
    main()