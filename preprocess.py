import numpy as np
import cv2


def bgr_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hsv_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def hsv_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def crop_roi(image, y1, y2, x1, x2):
    return image[y1:y2, x1:x2]

def resize(image, new_x=200, new_y=66):
    return cv2.resize(image, dsize=(new_x, new_y))

def threshold_steering(steer_value):
    if steer_value < -1:
        return -1
    if steer_value > 1:
        return 1
    return steer_value

def random_translation(image, steer_value, trans_range=40, steer_adjust=0.01):
    x_shift = trans_range * np.random.uniform() - trans_range/2
    M = np.float32([[1, 0, x_shift], [0, 1, 0]])
    new_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    new_steering = threshold_steering(steer_value + steer_adjust * x_shift)
    return new_image, new_steering

def random_rotation(image, steer_value, angle_range=20, steer_multiplier=1.5):
    r, c, _ = image.shape
    angle = np.random.uniform(angle_range) - angle_range/2
    M = cv2.getRotationMatrix2D((c/2, r/2), angle, 1.0)
    new_image = cv2.warpAffine(image, M, (c, r), flags=cv2.INTER_LINEAR)
    new_steering = threshold_steering(angle/90 * steer_multiplier) * -1
    return new_image, new_steering

def random_brightness(image, a=0.2, b=1.5, return_hsv=True):
    hsv = bgr_to_hsv(image)
    factor = np.random.uniform(a, b)
    print(factor)
    hsv[...,2] = np.minimum(hsv[...,2] * factor, 255)
    if return_hsv:
        return hsv
    return hsv_to_bgr(hsv)