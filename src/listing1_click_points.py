# Listing 1: Code for clicking a quadrangle (provided in the assignment).
import cv2
import numpy as np

N = 6
global n
n = 0

p1 = np.empty((N, 2))
p2 = np.empty((N, 2))

DISPLAY_W, DISPLAY_H = 900, 650


def draw_circle(event, x, y, flags, param):
    global n
    p = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param[1], (x, y), 5, (255, 0, 0), -1)
        p[n] = (x, y)
        n += 1


im1 = cv2.imread('a2_images/c1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
im2 = cv2.imread('a2_images/c2.jpg', cv2.IMREAD_REDUCED_COLOR_4)

im1copy = im1.copy()
im2copy = im2.copy()

cv2.namedWindow('Image 1', cv2.WINDOW_AUTOSIZE)
param = [p1, im1copy]
cv2.setMouseCallback('Image 1', draw_circle, param)

while True:
    cv2.imshow('Image 1', im1copy)
    if n == N:
        break
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyWindow('Image 1')

param = [p2, im2copy]
n = 0
cv2.namedWindow('Image 2', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Image 2', draw_circle, param)

while True:
    cv2.imshow('Image 2', im2copy)
    if n == N:
        break
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyWindow('Image 2')

print(p1)
print(p2)
