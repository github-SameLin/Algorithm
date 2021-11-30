import cv2
import math
import numpy as np
HSI = np.empty(shape=[3712,5568,3])
img  = cv2.imread("test.jpg")
img = img.astype(np.int32)
print(img[541][2514][0],img[541][2514][1],img[541][2514][2])
r= img[541][2514][0]
g=img[541][2514][1]
b=img[541][2514][2]
print(type(r))
print((2.0*r-g-b)/2.0)
print(((r-g)**2.0+((r-b)*(g-b))**0.5))
print(((r-b)*(g-b))**0.5)
print((r-b)*(g-b))
print(f'{r}\t{b}\t{r-b}')
print(g-b)
print(r,g,b)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        R = img[i][j][0]
        G = img[i][j][1]
        B = img[i][j][2]

        if R==G and G==B :
            theta = 0.0
        else:
            theta = math.acos(((2.0*R-G-B)/2.0)/((R-G)**2.0+((R-B)*(G-B))**0.5))
        # print(i,j,theta)
        if G>=B:
            H = theta
        else:
            H = theta-2.0*math.pi

        S = 1.0-(3.0/(R+G+B))*min(R,G,B)
        I = (R+G+B)/3.0
        # print(H,S,I)
        HSI[i][j][0] = H
        HSI[i][j][1] = S
        HSI[i][j][2] = I
print(HSI)
cv2.imshow(HSI)

