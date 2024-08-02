import os
import cv2
from cvzone.HandTrackingModule import HandDetector

# variables
width, height =1280, 720
folderPath = "presentation"

# camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of images
pathImages = sorted(os.listdir(folderPath),key=len)
# print(pathImages)

# variables
imgNumber = 0
hs, ws = int(120*1.2), int(213*1.2)

# HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # import images
    success, img = cap.read()
    img=cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath,pathImages[0])
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        print(fingers)

        

    # adding web cam image on slide
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall
    cv2.imshow("image", img)
    cv2.imshow("slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break