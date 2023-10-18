import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Track up to two hands

offset = 20
imgSize = 300

folder = "Data/Z"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        # Find a bounding box that encompasses both hands
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), -float('inf'), -float('inf')
        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x - offset)
            y_min = min(y_min, y - offset)
            x_max = max(x_max, x + w + offset)
            y_max = max(y_max, y + h + offset)

        # Ensure that the bounding box coordinates are within the image boundaries
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, img.shape[1])
        y_max = min(y_max, img.shape[0])

        # Crop the region containing both hands
        imgCrop = img[y_min:y_max, x_min:x_max]

        imgCropShape = imgCrop.shape

        aspectRatio = imgCropShape[1] / imgCropShape[0]

        if aspectRatio > 1:
            k = imgSize / imgCropShape[1]
            wCal = math.ceil(k * imgCropShape[1])
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / imgCropShape[0]
            hCal = math.ceil(k * imgCropShape[0])
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    
    if key == ord("q"):  # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
