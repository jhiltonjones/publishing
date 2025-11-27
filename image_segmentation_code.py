#Computes the angle between the 2 vectors by changing them into polar coordinates and computing the arctangent.
#Image segmentation finds the 2 red spots and then uses the angle method

import numpy as np 
import cv2
import matplotlib.pyplot as plt

def compute_signed_angle(v1,v2):
    angle1 = np.arctan2(v1[1], v2[0])
    angle2 = np.arctan2(v1[1], v2[0])
    angle_deg = np.degrees(angle1 - angle2)
    if angle_deg >180: angle_deg-=360
    elif angle_deg < -180: angle_deg +=360
    return angle_deg


def detect_red_points_and_angle(image_path, show=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image could not be read")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#makes colour thresholding more robust as it seperates chroma from brightness
    ranges = [(np.array([0,50,50]), np.array([10,255,255])),
              (np.array([160,50,50]), np.array([180,255,255]))]
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi) #makes a binary mask for each range if needed
        mask = m if mask is None else cv2.bitwise_or(mask,m) # this creates a mask where the red pixels are 255 and everythign else is 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError(f"Less than 2 points detected")
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:2] #gives the boundary points
    centers=[]
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))#computes the center of mass by diving the moment arm by the total mass of the contour
        pt1, pt2 = centers
        vec = np.array(pt2) - np.array(pt1)
        angle = compute_signed_angle(np.array(([1,0])), vec) 
        if show:
            cv2.line(image, pt1, pt2, (0,255,0), 2) #green line between the points
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)); plt.axis("off"); plt.show()
        return pt1, pt2, angle

