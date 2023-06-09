import cv2
import numpy as np
import matplotlib.pyplot as plt
#
imSize = []
clickedPoints = 0
points = []

# Function to handle mouse click events
def mouse_callback(event, x, y, flags, param):
    global clickedPoints, points,imSize
    xval = imSize[0] / 2
    yval = imSize[1] / 2
    if event == cv2.EVENT_LBUTTONDOWN:
        clickedPoints += 1
        points.append((((x)/100), (y/100), 1))
        # print("Clicked at position: ({}, {})".format(x, y))
        # print("Clicked at position in cebtered: ({}, {})".format(x - xval, -y + yval))


# Read the image using OpenCV
image = cv2.imread("img.png")
imSize = image.shape
print(imSize)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Image")

# Set the mouse callback function for the window
cv2.setMouseCallback("Image", mouse_callback)

# Display the image
cv2.imshow("Image", image)

# Wait until the desired number of clicks is reached or the window is closed
while clickedPoints < 4:
    cv2.waitKey(1)

# Close the window
cv2.destroyAllWindows()

#76 158
#113 138
#101 190
#140 163
# points[0] = (76/100, 158/100, 1)
# points[1] = (113/100, 138/100, 1)
# points[2] = (101/100, 190/100, 1)
# points[3] = (140/100, 163/100, 1)


droite1 = np.cross(points[0], points[1])
droite2 = np.cross(points[2], points[3])
f1 = np.cross(droite1, droite2)

droite3 = np.cross(points[1], points[3])
droite4 = np.cross(points[0], points[2])
f2 = np.cross(droite3, droite4)

hvals = np.cross(f1, f2)
norm = np.linalg.norm(hvals)
hvals /= norm
print(hvals)

H2 = np.array([[1,0,0], [0,1,0], [hvals[0], hvals[1], hvals[2]]])
#H2 = np.array([[ 1, 0, hvals[0]], [ 0, 1, hvals[1]], [ 0, 0, hvals[2]]])

result = np.zeros_like(image)
halfX = imSize[0] / 2
halfY = imSize[1] / 2

for y in range(imSize[0]):
    for x in range(imSize[1]):
        # Access the pixel value at (x, y)
        value = np.array([(x/100), (y/100), 1])
        r = np.dot(H2, value)
        r = np.abs(r)

        # xval = np.int32(np.floor((r[0] / r[2])))
        # yval = np.int32(np.floor((r[1] / r[2])))
        xval = np.int32((r[0] * 100)/r[2])
        yval = np.int32((r[1]*100)/r[2])

        # print((((r[0] / r[2])), ((r[1] / r[2]))))
        # print((xval, yval))
        # print(r)

        if xval < imSize[0] and yval < imSize[1] and xval > 0 and yval > 0:
            result[xval, yval, :] = image[x, y, :]

mask = (result == 0).all(axis=2)
image_filled = cv2.inpaint(result, mask.astype(np.uint8), 3, cv2.INPAINT_NS)

cv2.imshow("Image", result)

cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()