# GLaDOS Robot code.. 
# aka glados-home

This code assumes that you have 3 different systems
A pi 5 running 2 160 degree fish eye cameras
Zero Spy Camera for Raspberry Pi Zero - 160 Degree Focal Angle
https://www.adafruit.com/product/5390
and a voice bonnet
https://www.adafruit.com/product/4757
you will also require speakers that work with it
The pi5 also uses the stema connector on the voice bonnet for the following i2c modules

IMU https://www.adafruit.com/product/2472

SHt40 Temp and Humidity https://www.adafruit.com/product/4885

TOF VL53l4cx distance sensor https://www.adafruit.com/product/5425

ENS160 MOX Gas Sensor https://www.adafruit.com/product/5606

The pi5 is controlled by the GLaDOS.py script and its respective config file
Calling it looks like
python3 GLaDOS.py -c config.conf

The pi4 b+ handles the leds and body control movement
No all hardware is currently covered in the readme at this time

Current servos are

mg90d - Head up down
mg92b x 2 - Neck left & right, Body mid up down
gs3508mg - Main rotation Servo

Servo board is a 16 channel Servo Hat
https://www.adafruit.com/product/3416

GLaDOS eye is a 5mm Neo Pixel
https://www.adafruit.com/product/1938

2x 0.96 Inch 240*198 ST7789 Round Circular IPS S PI LCD Module Panel Display Screen for Arduino ESP32 Raspberry Pi STM32 CH32 C51
https://www.aliexpress.us/item/3256806072157291.html?spm=a2g0o.order_list.order_list_main.5.e4591802WmwKxj&gatewayAdapt=glo2usa

Running the body server on the PI4

python3 BodyServer.py -c config.conf

The AI server runs on a general purpose ubuntu server with a 4090 GPU
The python code requires CUDA support for the AI/ML Features
The AI server also requires MQTT to be installed and running in the background

Running the AI Server is as follows
python3 AiServer.py -c config.conf

Recommended startup order of the hardware due to cors connects is as follows

Start the BodyServer on the pi4
Then the AIServer on the ubuntu server
Finally run in the Main Glados code on the pi5



