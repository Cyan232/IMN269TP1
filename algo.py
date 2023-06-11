import cv2
import numpy as np
import matplotlib.pyplot as plt

imSize = []
clickedPoints = 0
points = []
H2 = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0, 0.0, 1.0]])

#Conversion des points en droites
def PointsToDroite(pts1, pts2):
    a = np.hstack((pts1, 1))
    b = np.hstack((pts2, 1))
    d = np.cross(a, b)
    d = d / np.linalg.norm(d) 
    return d

# Function pour capturer les points
def onMouse_Callback(event, x, y, flags, param):
    global clickedPoints, points,imSize
    if event == cv2.EVENT_LBUTTONDOWN:
        clickedPoints += 1
        points.append([x, y])

def onWheel_Callback(event, x, y, flags, param):
    global H2, imSize, image

    if event == cv2.EVENT_MOUSEWHEEL:
        scale_change = 0.1
        if flags > 0:
            H2[0, 0] += scale_change
            H2[1, 1] += scale_change
        else:
            H2[0, 0] -= scale_change
            H2[1, 1] -= scale_change

        result = np.zeros_like(image)
        result = cv2.warpPerspective(image, H2, (imSize[0], imSize[1]))
        cv2.imshow(param, result)

#Lecture de l'image
image = cv2.imread("img1.png")
imSize = image.shape
cv2.namedWindow("Image")

# Ajoute le callback pour choisir les points
cv2.setMouseCallback("Image", onMouse_Callback)

cv2.imshow("Image", image)

#Capture des points
while clickedPoints < 8:
    cv2.waitKey(1)

cv2.destroyAllWindows()

#Dessine les droites choisi
temp = cv2.line(image, points[0], points[1], (0,255,0), 2)
temp = cv2.line(image, points[2], points[3], (0,255,0), 2)
temp = cv2.line(image, points[4], points[5], (0,255,0), 2)
temp = cv2.line(image, points[6], points[7], (0,255,0), 2)

#Calculs des valeur pour trouver H_2
droite1 = PointsToDroite(points[0], points[1])
droite2 = PointsToDroite(points[2], points[3])
droite3 = PointsToDroite(points[4], points[5])
droite4 = PointsToDroite(points[6], points[7])

f1 = np.cross(droite1, droite2)
f2 = np.cross(droite3, droite4)

f1 /= f1[2]
f2 /= f2[2]

d = np.cross(f1, f2)
d /= d[2]

H2 = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0], [d[0], d[1], d[2]]])

#Vérification de H2
H2_inv = np.linalg.inv(H2)
H2_inv_transpose = H2_inv.T
infd = np.dot(H2_inv_transpose, d)
print(infd)

#On applique H2 a l'image
result = np.zeros_like(image)
result = cv2.warpPerspective(image, H2, [imSize[0], imSize[1]], result)

cv2.imshow("ImageWithLines", temp)
cv2.namedWindow("Final")
cv2.imshow("Final", result)
cv2.setMouseCallback("Final", onWheel_Callback, "Final")
cv2.waitKey(0)
cv2.destroyAllWindows()