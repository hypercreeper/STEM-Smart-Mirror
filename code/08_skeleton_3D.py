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
print("import threading")
import threading
print("import open3d")
import open3d as o3d
print("import numpy")
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from multiprocessing import Process
import os
from PIL import Image
import json
print("import flask")
from flask import Flask, request, Response

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='body', help=' Select mode: hand / body / holistic')
parser.add_argument('-c', '--camera', default='-1', help=' Input camera index')
parser.add_argument('-o', '--disable-mouse', default='0', help=' Disable Mouse Camera')
parser.add_argument('-a', '--disable-ai', default='0', help=' Disable AI')
args = parser.parse_args()
mode = args.mode
camera = args.camera
disablemouse = args.disable_mouse
disableai = args.disable_ai

print("utils display")
from utils_display import DisplayHand, DisplayBody
print("utils mediapipe")
from utils_mediapipe import MediaPipeHand, MediaPipeBody
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition
import pyautogui as pg
pg.FAILSAFE = True
# import cProfile

def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


# Start video capture
cap = False
if  camera == '-1':
    cap = cv2.VideoCapture(int(input("Enter Camera Index: "))) # By default webcam is index 0
else:
    cap = cv2.VideoCapture(int(camera)) # By default webcam is index 0
# cap = cv2.VideoCapture('../data/video.mp4') # Read from .mp4 file
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
print(img.shape)
# Load mediapipe and display class
    # Note: As of version 0.8.3 3D joint estimation is only available in full body mode
pipe = MediaPipeBody(static_image_mode=False, model_complexity=1, intrin=intrin,  enable_segmentation=False)
disp = DisplayBody(draw3d=False, draw_camera=True, intrin=intrin)

# log = False
# count = 0
# cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
    
# Load mediapipe class
pipegest = MediaPipeHand(static_image_mode=False, max_num_hands=1, model_complexity=1)

    # Load display class
dispgest = DisplayHand(max_num_hands=1)
    # Load gesture recognition class
gest = GestureRecognition('eval')
mousedown = False
def crop(img):
    return img[0:img.shape[1], 426:853]
def rotate(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # return img
def compute_angle_between_3d_points(a, b):
    # Compute the vector from point B to point A
    ba = a - b

    # Calculate the angle in radians
    return np.arctan2(ba[2], ba[0])

loaded_model_paths = {"shirt":[],"pants":[]}
loaded_images = {"shirt":[],"pants":[]}
for file in os.listdir("../data/clothes_models"):
    if file.endswith(".png"):
        loaded_model_paths[file[file.index("-")+1:file.index(".")]].append(os.path.abspath(os.path.join("../data/clothes_models",file)))
        loaded_images[file[file.index("-")+1:file.index(".")]].append(Image.open(os.path.join("../data/clothes_models",file)))
    elif file.endswith(".jpg"):
        loaded_model_paths[file[file.index("-")+1:file.index(".")]].append(os.path.abspath(os.path.join("../data/clothes_models",file)))
        loaded_images[file[file.index("-")+1:file.index(".")]].append(Image.open(os.path.join("../data/clothes_models",file)))

'''
clothing_model = o3d.io.read_triangle_mesh(loaded_model_paths[2])
disp.vis.add_geometry(clothing_model)
# disp.vis.draw_geometries(clothing_model)

RR = clothing_model.get_rotation_matrix_from_xyz((np.pi, 0, 0))
clothing_model.rotate(RR)
previous_degree = 0
'''
selectedGarment = {"shirt":0,"pants":0}
garmentUpdated = False
offset = 0
prev_time = time.time()
def start():
    global previous_degree, selectedGarment, garmentUpdated, clothing_model, loaded_model_paths, disp, loaded_images, offset, mousedown
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop back
            ret, img = cap.read()
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = crop(img)
        # img = rotate(img)
        # To improve performance, optionally mark image as not writeable to pass by reference
        img.flags.writeable = False
        if disablemouse != '1':
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
        img = crop(img)
        # img = rotate(img)
        # Flip image for 3rd person view
        img = cv2.flip(img, 1)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)

        # To improve performance, optionally mark image as not writeable to pass by reference
        img.flags.writeable = False
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
        if disablemouse != '1':
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
            cv2.imshow('img 2D', disp.draw2d(img, param, loaded_images["shirt"][selectedGarment["shirt"]], loaded_images["pants"][selectedGarment["pants"]], offset))
        else:
            cv2.imshow('img 2D', img)
        '''
        if garmentUpdated == True:
            print("clearing all geometries")
            # disp.clearAllExtObj()
            print("setting temp")
            temp_clothing_model = o3d.io.read_triangle_mesh(loaded_model_paths[selectedGarment])
            print("setting clothing_model variable")
            disp.vis.add_geometry(temp_clothing_model)
            RR = temp_clothing_model.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            temp_clothing_model.rotate(RR)
            clothing_model = temp_clothing_model
            garmentUpdated = False
        # Display 3D
        if disableai != '1':
            angle = compute_angle_between_3d_points(param["joint"][24], param["joint"][23])
            RR = clothing_model.get_rotation_matrix_from_xyz((0, np.deg2rad(previous_degree-np.rad2deg(angle)), 0))
            previous_degree = np.rad2deg(angle)
            clothing_model.rotate(RR)
            coord = param["joint"][24]
            coord[0] = (coord[0] + param["joint"][23][0])/2
            clothing_model.translate(coord, relative=False)
            distance = param["joint"][24][0] - param["joint"][23][0]
            clothing_model.scale(distance, (0,0,0))
            disp.vis.update_geometry(clothing_model)
            disp.vis.update_geometry(None)
            disp.vis.poll_events()
            disp.vis.update_renderer()
        disp.draw3d(param, img)
        # print(param["joint"])
                # Align clothing model to body keypoints
        # clothing_model.rotate(param['rotation'], center=(0,0,0))
        # clothing_model.translate(param['translation'])
        disp.vis.update_geometry(clothing_model)
        disp.vis.update_geometry(None)
        disp.vis.poll_events()
        disp.vis.update_renderer()
        '''
                
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
app = Flask(__name__)
@app.route('/startaiapp')
def start_ai_app():
    global disableai, disablemouse
    disableai = '0'
    return 'Starting...'
@app.route('/stopaiapp')
def stop_ai_app():
    global disableai, disablemouse
    disableai = '1'
    return 'Stopping...'
@app.route('/startmouse')
def start_ai_app():
    global disableai, disablemouse
    disablemouse = '0'
    return 'Starting...'
@app.route('/stopmouse')
def stop_ai_app():
    global disableai, disablemouse
    disablemouse = '1'
    return 'Stopping...'
@app.route('/selectgarment')
def select_garment():
    global loaded_model_paths, clothing_model, disp, selectedGarment, garmentUpdated
    i = int(request.headers["index"])
    type = request.headers["type"]
    if i < len(loaded_model_paths) and type in loaded_model_paths:
        selectedGarment[type] = i
        garmentUpdated = True
        return 'Selected ' + str(i) + ' in ' + type
    else:
        return Response('Index out of range or type invalid', 500)
@app.route('/getgarments')
def get_garments():
    global loaded_model_paths
    print(loaded_model_paths)
    return Response(json.dumps(loaded_model_paths), 200, mimetype="application/json")
@app.route('/setoffset')
def set_offset():
    global offset
    offset = int(request.headers["offset"])
    return "Offset Set"
def handleServer():
    server = Process(target=app.run(host='0.0.0.0'))
    server.start()
print("Starting Thread")
threading.Thread(target=handleServer).start()
print("Starting Main")
start()
print("shutting down server")
if disableai != '1':
    pipe.pipe.close()
if disablemouse != '1' and disableai != '1':
    pipegest.pipe.close()
cap.release()
