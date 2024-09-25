import cv2 as cv
import pytesseract

def get_text(img, threshold=150, max_value=255):
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    _, thresh = cv.threshold(gray, threshold, max_value, cv.THRESH_BINARY_INV)

    # Use PyTesseract to recognize text
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary
    text = pytesseract.image_to_string(thresh)
    
    return text

# Example usage
if __name__ == "__main__":
    # Path to the image
    img_path = 'text.jpg'

    # Load the image
    img = cv.imread(img_path)

    # Get the recognized text
    recognized_text = get_text(img)

    # Print the recognized text
    print("Recognized Text from sample:")
    print(recognized_text)

    # Read in sign images 
    sign1 = cv.imread('sign1.jpg')
    sign2 = cv.imread('sign2.jpg')
    sign3 = cv.imread('sign3.jpg')
    sign4 = cv.imread('sign4.jpg')

    # Get the recognized text from the sign images
    recognized_text1 = get_text(sign1)
    print("Recognized Text from sign1:")
    print(recognized_text1)