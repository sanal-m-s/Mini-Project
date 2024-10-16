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
imgNumber = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # Small image width and height

# Initialize zoom variables
zoomLevel = 1.0
targetZoomLevel = 1.0
zoomFactor = 0.1  # Amount to zoom in/out per gesture
zoomSpeed = 0.1  # Speed of zoom transition

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
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 0, 0), 10)

    if hands:  # If hand is detected
        hand = hands[0]
        cx, cy = hand["center"]
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        if cy <= gestureThreshold:  # If hand is at the height of the face
            # Gesture 1: To forward the slide
            if fingers == [1, 0, 0, 0, 0] and imgNumber > 0:
                imgNumber -= 1
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False

            # Gesture 2: To backward the slide
            if fingers == [0, 0, 0, 0, 1] and imgNumber < len(pathImages) - 1:
                imgNumber += 1
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False

            # Gesture 4: To mark on the slide
            if fingers == [0, 1, 0, 0, 0]:
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                annotations[annotationNumber].append((cx, cy))

            # Gesture 5: Zoom In and Zoom Out
            if fingers == [1, 1, 1, 1, 1]:
                targetZoomLevel += zoomFactor
            elif fingers == [0, 0, 0, 0, 0]:
                targetZoomLevel = max(1.0, targetZoomLevel - zoomFactor)

    # Smoothly interpolate the zoom level towards the target
    zoomLevel += (targetZoomLevel - zoomLevel) * zoomSpeed

    # Scale the current image based on zoomLevel
    new_width = int(imgCurrent.shape[1] * zoomLevel)
    new_height = int(imgCurrent.shape[0] * zoomLevel)
    imgCurrent = cv2.resize(imgCurrent, (new_width, new_height))

    # Create a blank canvas for displaying
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the cropping box to keep the image centered in the window
    offset_x = (new_width - width) // 2
    offset_y = (new_height - height) // 2

    # Ensure the cropping box does not go out of bounds
    imgCrop = imgCurrent[max(0, offset_y):offset_y + height, max(0, offset_x):offset_x + width]

    # Place the cropped image on the stable canvas
    canvas[0:height, 0:width] = imgCrop

    # Draw annotations
    for annotation in annotations:
        for i in range(1, len(annotation)):
            cv2.line(canvas, annotation[i - 1], annotation[i], (0, 0, 200), 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = canvas.shape
    canvas[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Slides", canvas)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
