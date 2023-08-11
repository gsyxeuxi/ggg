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
    val = 0
    incr = 10
    p1.start(val)
    p2.start(100)

    # print("PWM running. Press CTRL+C to exit.")
    # try:
    #     while True:
    #         time.sleep(0.25)
    #         if val >= 100:
    #             incr = -incr
    #         if val <= 0:
    #             incr = -incr
    #         val += incr
    #         p1.ChangeDutyCycle(val)
    #         p2.ChangeDutyCycle(val)
    # finally:
    #     p1.stop()
    #     p2.stop()
    #     GPIO.cleanup()
    
    
    # try:
    #     while True:
    #         time.sleep(200)
    # finally:
    #     p1.stop()
    #     p2.stop()
    #     GPIO.cleanup()



    while True:
        time.sleep(1)
        val2 = 100-val
        if val == 100:
            break
        val += incr
        p1.ChangeDutyCycle(val)
        p2.ChangeDutyCycle(val2)
    p1.stop()
    p2.stop()
    GPIO.cleanup()

if __name__ == '__main__':
    main()