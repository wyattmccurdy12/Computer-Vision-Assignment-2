# COS470 by Xin Zhang

import cv2
import numpy as np

# Load the image
image = cv2.imread('cards.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 75, 200)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare to draw contours
image_contours = image.copy()

# Loop over the contours
for contour in contours:
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # Perspective transform if the contour has 4 points (likely a card)
    if len(approx) == 4:
        # Draw contours
        cv2.drawContours(image_contours, [approx], -1, (0, 255, 0), 2)

        # Get the points for perspective transform
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Order the points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Set desired size and aspect ratio for the cards
        width = 200
        height = 300

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(image, M, (width, height))

        # Show the result
        cv2.imshow('Warped Card', warp)
        cv2.waitKey(0)

# Show the image with detected contours
cv2.imshow('Image with Contours', image_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
