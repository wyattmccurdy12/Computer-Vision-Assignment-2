'''
Wyatt McCurdy
COS 570 - Computer Vision
Dr. Zhang Xin
Assignment 2 part 1 - Measurements
'''

import cv2 as cv
import numpy as np

def main():
    # Load the image and convert to grayscale and blur
    image_path = 'book.jpg'
    img = cv.imread(image_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    # Edge detection
    edges = cv.Canny(blurred_img, 50, 120)

    # Find contours
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the outer contour (contours with no parent, i.e., hierarchy[i][3] == -1)
    outer_contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1]


    print(outer_contours)

    print("Contours found: ", len(contours))

    # Find the outer contour (assuming it's the largest contour)
    outer_contour = contours[0]

    # Find the inner contour (assuming it's the second largest contour)
    inner_contour = contours[2]

    # Approximate the contours to polygons
    epsilon_outer = 0.02 * cv.arcLength(outer_contour, True)
    outer_poly = cv.approxPolyDP(outer_contour, epsilon_outer, True)

    epsilon_inner = 0.02 * cv.arcLength(inner_contour, True)
    inner_poly = cv.approxPolyDP(inner_contour, epsilon_inner, True)

    # Draw the approximated polygons on the original image
    cv.drawContours(img, [outer_poly], -1, (0, 255, 0), 2)
    cv.drawContours(img, [inner_poly], -1, (255, 0, 0), 2)

    # Calculate the real width and height of the inner contour polygon
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    # Calculate the width and height based on the distances between the points
    outer_width_pixels = euclidean_distance(outer_poly[0][0], outer_poly[1][0])
    outer_height_pixels = euclidean_distance(outer_poly[0][0], outer_poly[3][0])

    inner_width_pixels = euclidean_distance(inner_poly[0][0], inner_poly[1][0])
    inner_height_pixels = euclidean_distance(inner_poly[0][0], inner_poly[3][0])

    # Assuming the outer contour represents a paper with width 21.5 cm and height 27.8 cm
    outer_width_cm = 21.5
    outer_height_cm = 27.8

    inner_width_cm = (inner_width_pixels / outer_width_pixels) * outer_width_cm
    inner_height_cm = (inner_height_pixels / outer_height_pixels) * outer_height_cm

    print(f"Inner Width: {inner_width_cm:.2f} cm")
    print(f"Inner Height: {inner_height_cm:.2f} cm")

    # Calculate the error
    actual_width_cm = 8.0
    actual_height_cm = 10.6

    width_error = abs(inner_width_cm - actual_width_cm)
    height_error = abs(inner_height_cm - actual_height_cm)

    width_error_percent = (width_error / actual_width_cm) * 100
    height_error_percent = (height_error / actual_height_cm) * 100

    print(f"Width Error: {width_error:.2f} cm ({width_error_percent:.2f}%)")
    print(f"Height Error: {height_error:.2f} cm ({height_error_percent:.2f}%)")

    # Annotate the width, height, and error on the image
    font_scale = 0.7
    thickness = 2
    cv.putText(img, f'Outer: {outer_width_cm:.2f} cm x {outer_height_cm:.2f} cm', (outer_poly[0][0][0] - 100, outer_poly[0][0][1] - 10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    cv.putText(img, f'Inner: {inner_width_cm:.2f} cm x {inner_height_cm:.2f} cm', (inner_poly[0][0][0], inner_poly[0][0][1] - 10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
    cv.putText(img, f'Width Error: {width_error:.2f} cm ({width_error_percent:.2f}%)', (10, 30), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv.putText(img, f'Height Error: {height_error:.2f} cm ({height_error_percent:.2f}%)', (10, 60), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Save the annotated image
    cv.imwrite('book_measurements.jpg', img)

    # Display the annotated image
    cv.imshow('Annotated Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()