import board
from time import sleep, time
import neopixel
import ledhelper
import busio
import adafruit_pca9685

from threading import Thread

class ledHead:
    def __init__(self):
        # note the LED in the eye is GRB not RGB make sure to convert
        self.pixels = neopixel.NeoPixel(board.D18, 1, brightness=1, auto_write=True)
        self.lh = ledhelper.LedHelper
        self.ani = ledhelper.NeoPixelAnimations(self.pixels, 1)
        self.swap = self.lh.rgb2grbswap
        # power led
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA))
        self.pwmled = self.hat.channels[4]
        self.hat.frequency = 60
        self.pwmled.duty_cycle= 250 
        # self.anger is a tuple which represents the major and minor anger, first being major, second being minor
        self.intensity = (.1, .1)

    def startup(self):
        # Do a startup sequence plusing the eye and head power LED from low to high...
        eyeledthread = Thread(target = self.ani.intensity, args=(10, self.swap((255, 255, 0))))
        pwmledthread = Thread(target = self.ani.pwmintensity, args=(10, self.pwmled))
        eyeledthread.start()
        pwmledthread.start()
        eyeledthread.join()
        pwmledthread.join()
        self.pwmled.duty_cycle= 150 
        self.pixels.brightness = self.intensity[0]
        self.pixels.autowrite = True
        self.pixels[0] = self.lh.adjust_brightness(self.swap((255,255, 0)), self.intensity[1])
        self.pixels.show()
    
    def disco(self):
        # set intensity to half
        self.intensity = (.8, .8)
        self.pixels.brightness = self.intensity[0]
        eyeledthread = Thread(target = self.ani.rainbow_cycle, args = (.05, "GRB"))
        pwmledthread = Thread(target = self.ani.pwmintensity, args = (10, self.pwmled))
        eyeledthread.start()
        pwmledthread.start()
        eyeledthread.join()
        pwmledthread.join()
   
   #TODO you left off considering how to handle intensity across the entire robot 
    def angryeye(self, steps= 20, very_angry = True):
        self.intensity = (.1, .1)
        self.pixels.brightness = self.intensity[0]
        self.pixels[0] = (255, 255, 0)
        self.pixels.show()
        sleep(1.4)
        anger = (255, 69, 0)
        if very_angry is True:
            anger = (139, 0, 0)
            self.pwmled.duty_cycle= 65535
            self.intensity = (0.9, 0.9)
        self.pixels.brightness = self.intensity[0]
        eyeledthread = Thread(target = self.ani.fade_color, args = ((255,255,0), anger, steps, "GRB", self.intensity))
        eyeledthread.start()
        eyeledthread.join()


if __name__ == "__main__":
    lh = ledHead()
    lh.startup()
    #lh.disco()
    #lh.angryeye(very_angry=True)
        

