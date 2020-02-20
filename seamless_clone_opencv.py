import cv2
import numpy as np
# Read images : src image will be cloned into dst
dst = cv2.imread("glare.png")
obj = cv2.imread("no_glare_noise.png")
mask = cv2.imread("mask.png")[:,:,0] / 255.0

# Create an all white mask
# mask = np.ones(obj.shape, obj.dtype)
# The location of the center of the src in the dst
width, height, channels = dst.shape
center = (int(height/2), int(width/2))
# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, dst, mask,  center, cv2.MIXED_CLONE)
# Write results
cv2.imwrite("opencv-normal-clone-example.jpg", normal_clone)
cv2.imwrite("opencv-mixed-clone-example.jpg", mixed_clone)