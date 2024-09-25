import cv2 as cv
import pytesseract as pt
import numpy as np
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary


def preprocess_image(image):
    """
    Preprocess the image by converting to grayscale and applying Gaussian blur.
    
    Args:
    image (ndarray): Input image.
    
    Returns:
    blurred_img (ndarray): Preprocessed grayscale and blurred image.
    """
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    return blurred_img

def get_contours(image):
    """
    Get contours from the preprocessed image.
    
    Args:
    image (ndarray): Preprocessed image.
    
    Returns:
    contours (list): List of contours found in the image.
    """
    edges = cv.Canny(image, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

# Example usage
if __name__ == "__main__":
    # Path to the image
    img_path = 'text.jpg'

    # Load the image
    img = cv.imread(img_path)

    # Get the recognized text
    recognized_text = pt.image_to_string(img)

    # Print the recognized text
    print("Recognized Text from sample:")
    print(recognized_text)

    # Read in sign images 
    sign1 = cv.imread('sign1.jpg')
    sign2 = cv.imread('sign2.jpg')
    sign3 = cv.imread('sign3.jpg')
    sign4 = cv.imread('sign4.jpg')

    # Preprocess the first sign image
    preprocessed_sign1 = preprocess_image(sign1)

    # Get contours for the first sign image
    contours = get_contours(preprocessed_sign1)

    # Draw contours on the original sign image
    cv.drawContours(sign1, contours, -1, (0, 255, 0), 2)

    # Display the first sign image with contours
    cv.imshow('Sign 1 with Contours', sign1)
    cv.waitKey(0)
    cv.destroyAllWindows()