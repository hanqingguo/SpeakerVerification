import cv2
import numpy as np

img_rgb = cv2.imread("./mfcc-img/jianzhi3.png")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('./mfcc-img/hanqing-t.PNG',0)

w, h = template.shape[::-1]
print(w, h)
# cv2.imshow('rgb',img_rgb)
# cv2.imshow('gray',img_gray)
# cv2.imshow('template',template)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 2)
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()