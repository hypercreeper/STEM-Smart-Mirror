###############################################################################
### Simple demo on displaying 3D hand/body skeleton
### Input : Live video of hand/body
### Output: 3D display of hand/body skeleton 
### Usage : python 08_skeleton_3D.py -m hand
###       : python 08_skeleton_3D.py -m body
###       : python 08_skeleton_3D.py -m holistic
###############################################################################
print("start program")
print("import cv2")
import cv2
print("import time")
import time
print("import argparse")
import argparse
# import numpy as np
print("utils display")
from utils_display import DisplayHand, DisplayBody, DisplayHolistic
print("utils mediapipe")
from utils_mediapipe import MediaPipeHand, MediaPipeBody, MediaPipeHolistic
from utils_joint_angle import GestureRecognition
import pyautogui as pg
# import cProfile

def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='hand', help=' Select mode: hand / body / holistic')
parser.add_argument('-c', '--camera', default='-1', help=' Input camera index')
parser.add_argument('-o', '--disable-mouse', default='0', help=' Disable Mouse Camera')
parser.add_argument('-a', '--disable-ai', default='0', help=' Disable AI')
args = parser.parse_args()
mode = args.mode
camera = args.camera
disablemouse = args.disable_mouse
disableai = args.disable_ai

# Start video capture
cap = False
if  camera == '-1':
    cap = cv2.VideoCapture(int(input("Enter Camera Index: "))) # By default webcam is index 0
else:
    cap = cv2.VideoCapture(int(camera)) # By default webcam is index 0
# cap = cv2.VideoCapture('../data/video.mp4') # Read from .mp4 file

# Read in sample image to estimate camera intrinsic
ret, img = cap.read(0)
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
# print(dir(img))
img_width  = img.shape[1]
img_height = img.shape[0]
intrin = {
    'fx': img_width*0.9, # Approx 0.7w < f < w https://www.learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
    'fy': img_width*0.9,
    'cx': img_width*0.5, # Approx center of image
    'cy': img_height*0.5,
    'width': img_width,
    'height': img_height,
}

if disableai != '1':
# Load mediapipe and display class
    if mode=='hand':
        pipe = MediaPipeHand(static_image_mode=False, max_num_hands=2, intrin=intrin)
        disp = DisplayHand(draw3d=True, draw_camera=True, max_num_hands=2, intrin=intrin)
    elif mode=='body':
        # Note: As of version 0.8.3 3D joint estimation is only available in full body mode
        pipe = MediaPipeBody(static_image_mode=False, model_complexity=1, intrin=intrin,  enable_segmentation=False)
        disp = DisplayBody(draw3d=False, draw_camera=True, intrin=intrin)
    elif mode=='holistic':
        # Note: As of version 0.8.3 3D joint estimation is only available in full body mode
        pipe = MediaPipeHolistic(static_image_mode=False, model_complexity=1, intrin=intrin)
        disp = DisplayHolistic(draw3d=True, draw_camera=True, intrin=intrin)

# log = False
# count = 0
# cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
    
# Load mediapipe class
if disablemouse != '1' or disableai != '1' :
    pipegest = MediaPipeHand(static_image_mode=False, max_num_hands=1, model_complexity=1)

    # Load display class
    dispgest = DisplayHand(max_num_hands=1)
    # Load gesture recognition class
    gest = GestureRecognition('eval')
mousedown = False
prev_time = time.time()
def start():
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back
            ret, img = cap.read()

        # To improve performance, optionally mark image as not writeable to pass by reference
        img.flags.writeable = False
        if disablemouse != '1' or disableai != '1':
        # Feedforward to extract keypoint
            paramgest = pipegest.forward(img)
            if (paramgest[0]['class'] is not None):
                paramgest[0]['gesture'] = gest.eval(paramgest[0]['angle'])

        img.flags.writeable = True

        # Display keypoint
        # cv2.imshow('gesture 2D', dispgest.draw2d(img.copy(), paramgest))

        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back
            ret, img = cap.read()
            # break

        # Flip image for 3rd person view
        img = cv2.flip(img, 1)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)

        # To improve performance, optionally mark image as not writeable to pass by reference
        img.flags.writeable = False
        if disableai != '1':
        # Feedforward to extract keypoint
            param = pipe.forward(img)

        # Compute FPS
        # curr_time = time.time()
        # fps = 1/(curr_time-prev_time)
        # if mode=='body':
        #     param['fps'] = fps
        # elif mode=='face' or mode=='hand':
        #     param[0]['fps'] = fps
        # elif mode=='holistic':
        #     for p in param:
        #         p['fps'] = fps
        # prev_time = curr_time    

        img.flags.writeable = True
        if disablemouse != '1' or disableai != '1':
            x = int(param['keypt'][19,0])
            y = int(param['keypt'][19,1])
            # print(type(img_width))
            # print(type(pg.size().width))
            xx = map_range(int(x), int(0), int(img_width), int(0), int(pg.size().width))
            yy = map_range(int(y), int(0), int(img_height), int(0), int(pg.size().height))
            # print(xx)
            # print(yy)
            pg.moveTo(xx, yy)
            if paramgest[0]['gesture'] is not None:
                if(paramgest[0]['gesture'].lower() == "select"):
                    if mousedown is not True:
                        mousedown = True
                        pg.mouseDown()

                else:
                    mousedown = False
                    pg.mouseUp()
            # Display keypoint
        cv2.namedWindow('img 2D', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('img 2D', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if disableai != '1':
            cv2.imshow('img 2D', disp.draw2d(img, param))
        else:
            cv2.imshow('img 2D', img)
        # Display 3D
        # disp.draw3d(param, img)
        # disp.vis.update_geometry(None)
        # disp.vis.poll_events()
        # disp.vis.update_renderer()    

        # if log:
        #     img = (np.asarray(disp.vis.capture_screen_float_buffer())*255).astype(np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('../data/image/'+str(count).zfill(2)+'.png', img)
        #     count += 1

        key = cv2.waitKey(1)
        if key==27:
            break
        if key==ord('r'): # Press 'r' to reset camera view
            disp.camera.reset_view()
        # if key==32: # Press spacebar to start logging images
        #     log = not log
        #     print('Log', log)
# cProfile.run('start()','profileing')
start()
if disableai != '1':
    pipe.pipe.close()
if disablemouse != '1' or disableai != '1':
    pipegest.pipe.close()
cap.release()
