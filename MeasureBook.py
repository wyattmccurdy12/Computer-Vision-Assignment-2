'''
Wyatt McCurdy
COS 570 - Computer Vision
Dr. Zhang Xin
Assignment 2 part 1 - Measurements
'''

import cv2 as cv
import numpy as np

def load_and_preprocess_image(image_path):
    """
    Load the image, convert to grayscale, and apply Gaussian blur.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    img (ndarray): Original image.
    blurred_img (ndarray): Preprocessed grayscale and blurred image.
    """
    img = cv.imread(image_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    return img, blurred_img

def find_outer_contour(blurred_img):
    """
    Find the outer contour in the preprocessed image.
    
    Args:
    blurred_img (ndarray): Preprocessed grayscale and blurred image.
    
    Returns:
    outer_contour (ndarray): The largest outer contour.
    """
    edges = cv.Canny(blurred_img, 50, 120)  # Perform edge detection
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    outer_contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1]  # Filter outer contours
    outer_contour = max(outer_contours, key=cv.contourArea)  # Get the largest outer contour
    return outer_contour, outer_contours

def get_perspective_transform_points(contour):
    """
    Get the points for perspective transform.
    
    Args:
    contour (ndarray): Contour points.
    
    Returns:
    rect (ndarray): Ordered points for perspective transform.
    """
    peri = cv.arcLength(contour, True)  # Calculate the perimeter of the contour
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)  # Approximate the contour to a polygon
    pts = approx.reshape(4, 2)  # Reshape the points to a 4x2 array
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point
    rect[2] = pts[np.argmax(s)]  # Bottom-right point
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right point
    rect[3] = pts[np.argmax(diff)]  # Bottom-left point
    return rect

def warp_image(img, rect, width, height):
    """
    Warp the image using perspective transform.
    
    Args:
    img (ndarray): Original image.
    rect (ndarray): Points for perspective transform.
    width (int): Desired width of the warped image.
    height (int): Desired height of the warped image.
    
    Returns:
    warp (ndarray): Warped image.
    """
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")  # Destination points for perspective transform
    M = cv.getPerspectiveTransform(rect, dst)  # Compute the perspective transform matrix
    warp = cv.warpPerspective(img, M, (width, height))  # Apply the perspective transform
    return warp

def find_inner_contour(warp):
    """
    Find the inner contour in the warped image.
    
    Args:
    warp (ndarray): Warped image.
    
    Returns:
    inner_contour (ndarray): The largest inner contour.
    """
    warp_gray = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)  # Convert the warped image to grayscale
    warp_edges = cv.Canny(warp_gray, 50, 120)  # Perform edge detection
    warp_contours, _ = cv.findContours(warp_edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    inner_contour = max(warp_contours, key=cv.contourArea)  # Get the largest inner contour
    return inner_contour

def calculate_dimensions(rect):
    """
    Calculate the width and height of the rectangle in pixels and cm.
    
    Args:
    rect (ndarray): Points of the rectangle.
    
    Returns:
    width_cm (float): Width of the rectangle in cm.
    height_cm (float): Height of the rectangle in cm.
    """
    width_pix = np.linalg.norm(rect[0] - rect[1])  # Calculate the width in pixels
    height_pix = np.linalg.norm(rect[1] - rect[2])  # Calculate the height in pixels
    width_cm = width_pix / 40  # Convert width to cm (assuming 40 pixels per cm)
    height_cm = height_pix / 40  # Convert height to cm (assuming 40 pixels per cm)
    return width_cm, height_cm

def calculate_percent_error(measured, actual):
    """
    Calculate the percent error between measured and actual values.
    
    Args:
    measured (float): Measured value.
    actual (float): Actual value.
    
    Returns:
    percent_error (float): Percent error.
    """
    percent_error = ((measured - actual) / actual) * 100  # Calculate percent error
    return percent_error

def annotate_image(warp, rect, width_cm, height_cm, width_percent_error, height_percent_error):
    """
    Annotate the width, height, and percent error on the image.
    
    Args:
    warp (ndarray): Warped image.
    rect (ndarray): Points of the rectangle.
    width_cm (float): Width of the rectangle in cm.
    height_cm (float): Height of the rectangle in cm.
    width_percent_error (float): Percent error for width.
    height_percent_error (float): Percent error for height.
    """
    mid_width = (rect[0] + rect[1]) / 2  # Midpoint of the top edge
    mid_height = (rect[1] + rect[2]) / 2  # Midpoint of the right edge

    # Annotate width and height on the image
    cv.putText(warp, f"Width: {width_cm:.2f} cm", (int(mid_width[0]), int(mid_width[1])),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv.putText(warp, f"Height: {height_cm:.2f} cm", (int(mid_height[0]), int(mid_height[1])),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Annotate percent error in the top left corner
    cv.putText(warp, f"Width Error: {width_percent_error:.2f}%", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv.putText(warp, f"Height Error: {height_percent_error:.2f}%", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

def main():
    # Load and preprocess the image
    image_path = 'book.jpg'
    img, blurred_img = load_and_preprocess_image(image_path)

    # Find the outer contour
    outer_contour, outer_contours = find_outer_contour(blurred_img)

    # Draw the outer contours on the original image
    cv.drawContours(img, outer_contours, -1, (0, 255, 0), 2)

    # Get the points for perspective transform
    rect = get_perspective_transform_points(outer_contour)

    # Set desired size and aspect ratio for the warped image
    width = 278 * 4
    height = 215 * 4

    # Warp the image
    warp = warp_image(img, rect, width, height)

    # Find the inner contour in the warped image
    inner_contour = find_inner_contour(warp)
    cv.drawContours(warp, [inner_contour], -1, (255, 0, 0), 2)  # Draw inner contour in blue

    # Get the points for perspective transform of the inner contour
    rect = get_perspective_transform_points(inner_contour)

    # Calculate the dimensions of the inner contour rectangle
    width_cm, height_cm = calculate_dimensions(rect)

    # Actual dimensions in cm
    actual_width_cm = 8
    actual_height_cm = 10.6

    # Calculate the percent error
    width_percent_error = calculate_percent_error(width_cm, actual_width_cm)
    height_percent_error = calculate_percent_error(height_cm, actual_height_cm)

    # Annotate the image with dimensions and percent error
    annotate_image(warp, rect, width_cm, height_cm, width_percent_error, height_percent_error)

    # Display the original image with outer contours
    cv.imshow('Original Image with Outer Contours', img)

    # Display the warped image with inner contour and annotations
    cv.imshow('Warped Image with Inner Contour', warp)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()