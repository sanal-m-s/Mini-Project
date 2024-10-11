import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector
# Parameters
width, height = 1280, 720
gestureThreshold = 300
folderPath = "Presentation"
# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)
# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image
# Initialize zoom variables
zoomLevel = 1.0
zoomFactor = 0.1  # Amount to zoom in/out per gesture
# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)
while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 0, 0), 10)
    if hands and buttonPressed is False:  # If hand is detected
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up
        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal
        if cy <= gestureThreshold:  # If hand is at the height of the face

            # Gesture 1:To forward the slide
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

            # Gesture 2:To backward the slide
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False


            # Gesture 3:To navigate the slide
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            # Gesture 4:To mark on the slide
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        else:
            annotationStart = False

            # Gesture 5 for Zoom In
            # Inside your while loop where you handle gestures
            if hands and buttonPressed is False:  # If hand is detected
                hand = hands[0]
                cx, cy = hand["center"]  # Center of the hand

                # Gesture for Zoom In
                if fingers == [1, 1, 1, 1, 1]:
                    zoomLevel += zoomFactor
                    print("Zoom In")

                # Gesture for Zoom Out
                if fingers == [0, 0, 0, 0, 0]:
                    zoomLevel = max(1.0, zoomLevel - zoomFactor)
                    print("Zoom Out")

                # Scale the current image based on zoomLevel
            new_width = int(imgCurrent.shape[1] * zoomLevel)
            new_height = int(imgCurrent.shape[0] * zoomLevel)
            imgCurrent = cv2.resize(imgCurrent, (new_width, new_height))

            # Calculate the new position to keep the finger centered
            offset_x = int(cx * zoomLevel) - (width // 2)
            offset_y = int(cy * zoomLevel) - (height // 2)

            # Ensure the image doesn't exceed display bounds
            imgCurrent = imgCurrent[max(0, offset_y):min(new_height, offset_y + height),
                         max(0, offset_x):min(new_width, offset_x + width)]

            # Display the image
            cv2.imshow("Slides", imgCurrent)

            # ... (remaining code) ...

            # Gesture 7:To undo the marked part
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True
    else:
        annotationStart = False
    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False
    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall
    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
