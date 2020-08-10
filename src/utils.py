import numpy as np
import cv2

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50,thickness=(2,2,2)):
    """
    Function used to draw y (headpose label) on Input Image x.
    Implemented by: shamangary
    https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
    Modified by: Omar Hassan
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),thickness[2])

    return img
